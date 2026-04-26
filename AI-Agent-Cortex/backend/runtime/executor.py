"""Individual agent executor using LangChain + OpenAI."""
import json
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from config import settings
from runtime.tools import execute_tool, get_tools_for_agent

logger = logging.getLogger(__name__)

# Cost per 1K tokens (gpt-4o-mini pricing)
COST_PER_1K_INPUT = 0.00015
COST_PER_1K_OUTPUT = 0.0006


class AgentExecutor:
    """Executes a single agent's logic: LLM call with tools and memory."""

    def __init__(self, agent_config: dict):
        self.agent_id = agent_config["id"]
        self.name = agent_config["name"]
        self.role = agent_config.get("role", "assistant")
        self.system_prompt = agent_config.get("system_prompt", "You are a helpful assistant.")
        self.model_name = agent_config.get("model", settings.openai_model)
        self.tool_names = agent_config.get("tools", [])
        self.guardrails = agent_config.get("guardrails", {})
        self.memory: list[dict] = []
        self.total_tokens = 0
        self.total_cost = 0.0

        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=settings.openai_api_key,
            temperature=0.7,
            max_tokens=self.guardrails.get("max_tokens_per_response", 4096),
        )

    def _build_tool_instructions(self) -> str:
        """Build tool usage instructions for the system prompt."""
        if not self.tool_names:
            return ""

        tools = get_tools_for_agent(self.tool_names)
        if not tools:
            return ""

        tool_text = "\n\nYou have access to the following tools:\n"
        for tool in tools:
            tool_text += f"- **{tool['name']}**: {tool['description']}\n"
        tool_text += (
            "\nTo use a tool, include in your response: "
            '<tool_call>{"tool": "tool_name", "args": {"param": "value"}}</tool_call>\n'
            "You will receive the tool's output and can continue your response."
        )
        return tool_text

    async def execute(self, input_text: str, context: Optional[str] = None) -> dict:
        """Execute the agent with input and optional context from other agents."""
        # Build messages
        system_content = self.system_prompt + self._build_tool_instructions()
        if context:
            system_content += f"\n\nContext from other agents:\n{context}"

        messages = [SystemMessage(content=system_content)]

        # Add memory/history
        for mem in self.memory[-10:]:  # Last 10 messages
            if mem["role"] == "user":
                messages.append(HumanMessage(content=mem["content"]))
            else:
                messages.append(AIMessage(content=mem["content"]))

        messages.append(HumanMessage(content=input_text))

        # Check guardrails
        max_tokens_per_minute = self.guardrails.get("max_tokens_per_minute", 100000)
        if self.total_tokens > max_tokens_per_minute:
            return {
                "content": "Token limit exceeded. Please wait before sending more requests.",
                "tokens_used": 0,
                "cost": 0.0,
                "tool_calls": [],
            }

        # Call LLM
        try:
            response = await self.llm.ainvoke(messages)
            content = response.content

            # Track tokens
            input_tokens = response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0)
            output_tokens = response.response_metadata.get("token_usage", {}).get("completion_tokens", 0)
            tokens_used = input_tokens + output_tokens
            cost = (input_tokens / 1000 * COST_PER_1K_INPUT) + (output_tokens / 1000 * COST_PER_1K_OUTPUT)

            self.total_tokens += tokens_used
            self.total_cost += cost

            # Process tool calls in the response
            tool_results = []
            import re
            tool_pattern = r'<tool_call>(.*?)</tool_call>'
            tool_matches = re.findall(tool_pattern, content, re.DOTALL)

            for match in tool_matches:
                try:
                    tool_call = json.loads(match)
                    tool_name = tool_call.get("tool", "")
                    tool_args = tool_call.get("args", {})

                    if tool_name in self.tool_names:
                        result = await execute_tool(tool_name, **tool_args)
                        tool_results.append({
                            "tool": tool_name,
                            "args": tool_args,
                            "result": result,
                        })
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse tool call: {e}")

            # If tools were called, do a follow-up LLM call with results
            if tool_results:
                tool_context = "\n".join(
                    f"Tool '{r['tool']}' returned: {r['result']}" for r in tool_results
                )
                messages.append(AIMessage(content=content))
                messages.append(HumanMessage(
                    content=f"Tool results:\n{tool_context}\n\nPlease continue with your response using these results."
                ))

                response2 = await self.llm.ainvoke(messages)
                content = response2.content

                input_tokens2 = response2.response_metadata.get("token_usage", {}).get("prompt_tokens", 0)
                output_tokens2 = response2.response_metadata.get("token_usage", {}).get("completion_tokens", 0)
                tokens_used += input_tokens2 + output_tokens2
                cost += (input_tokens2 / 1000 * COST_PER_1K_INPUT) + (output_tokens2 / 1000 * COST_PER_1K_OUTPUT)

            # Update memory
            self.memory.append({"role": "user", "content": input_text})
            self.memory.append({"role": "assistant", "content": content})

            # Content filter guardrail
            blocked_keywords = self.guardrails.get("blocked_keywords", [])
            if blocked_keywords:
                for keyword in blocked_keywords:
                    if keyword.lower() in content.lower():
                        content = "[Content filtered by guardrail]"
                        break

            return {
                "content": content,
                "tokens_used": tokens_used,
                "cost": cost,
                "tool_calls": tool_results,
            }

        except Exception as e:
            logger.error(f"Agent {self.name} execution error: {e}")
            return {
                "content": f"Error executing agent: {str(e)}",
                "tokens_used": 0,
                "cost": 0.0,
                "tool_calls": [],
                "error": str(e),
            }
