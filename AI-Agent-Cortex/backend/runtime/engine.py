"""Workflow execution engine using LangGraph for stateful orchestration."""
import asyncio
import datetime
import json
import logging
from typing import Any, Optional

from langgraph.graph import StateGraph, END
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.database import async_session
from database.models import Agent, Workflow, WorkflowExecution, Message, generate_uuid
from runtime.executor import AgentExecutor

logger = logging.getLogger(__name__)


class WorkflowState(dict):
    """State that flows through the workflow graph."""
    pass


class WorkflowEngine:
    """Orchestrates multi-agent workflows using LangGraph."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def run(self, execution_id: str, workflow: Workflow, input_data: dict):
        """Run a workflow execution."""
        try:
            async with async_session() as db:
                # Load all agents for this workflow
                agent_configs = {}
                for agent_id in workflow.agents:
                    result = await db.execute(select(Agent).where(Agent.id == agent_id))
                    agent = result.scalar_one_or_none()
                    if agent:
                        agent_configs[agent_id] = {
                            "id": agent.id,
                            "name": agent.name,
                            "role": agent.role,
                            "system_prompt": agent.system_prompt,
                            "model": agent.model,
                            "tools": agent.tools or [],
                            "guardrails": agent.guardrails or {},
                        }

                graph_def = workflow.graph
                nodes = graph_def.get("nodes", [])
                edges = graph_def.get("edges", [])

                # Build role-to-agent mapping
                role_to_agent_id = {}
                for node in nodes:
                    if node["type"] == "agent":
                        config = node.get("config", {})
                        agent_id = config.get("agent_id")
                        role = config.get("role", "")
                        if agent_id and agent_id in agent_configs:
                            role_to_agent_id[role] = agent_id
                        elif role:
                            # Try to find agent by role match
                            for aid, aconf in agent_configs.items():
                                if aconf["role"] == role:
                                    role_to_agent_id[role] = aid
                                    break

                # Create executors for each agent
                executors: dict[str, AgentExecutor] = {}
                for agent_id, config in agent_configs.items():
                    executors[agent_id] = AgentExecutor(config)

                # Execute workflow step by step following the graph
                state = {
                    "input": input_data.get("query", input_data.get("input", str(input_data))),
                    "messages": [],
                    "current_output": "",
                    "iteration": 0,
                    "max_iterations": 10,
                    "status": "running",
                }

                # Find start node and trace execution path
                execution_order = self._resolve_execution_order(nodes, edges)
                total_tokens = 0
                total_cost = 0.0
                final_output = ""

                for step in execution_order:
                    if state["iteration"] >= state["max_iterations"]:
                        logger.warning(f"Workflow reached max iterations ({state['max_iterations']})")
                        break

                    node = next((n for n in nodes if n["id"] == step), None)
                    if not node:
                        continue

                    if node["type"] == "start":
                        await self._log_message(
                            db, execution_id, None, None,
                            f"Workflow started with input: {state['input']}",
                            "system"
                        )
                        continue

                    if node["type"] == "end":
                        await self._log_message(
                            db, execution_id, None, None,
                            f"Workflow completed",
                            "system"
                        )
                        break

                    if node["type"] == "condition":
                        # Evaluate condition based on current output
                        condition_result = self._evaluate_condition(
                            node, state["current_output"], edges
                        )
                        state["messages"].append({
                            "node": step,
                            "type": "condition",
                            "result": condition_result,
                        })

                        # If condition leads to a feedback loop, adjust execution order
                        if condition_result == "revision_needed":
                            # Find the target of the revision edge
                            for edge in edges:
                                if edge["source"] == step and edge.get("condition") == "revision_needed":
                                    revision_target = edge["target"]
                                    # Re-add the path from revision target onwards
                                    idx = execution_order.index(step)
                                    # Find remaining path from revision target
                                    for i, n in enumerate(execution_order):
                                        if n == revision_target:
                                            remaining = execution_order[i:idx+1]
                                            execution_order.extend(remaining)
                                            break
                                    break

                        await self._log_message(
                            db, execution_id, None, None,
                            f"Condition '{node['label']}': {condition_result}",
                            "condition"
                        )
                        continue

                    if node["type"] == "agent":
                        config = node.get("config", {})
                        role = config.get("role", "")
                        agent_id = config.get("agent_id") or role_to_agent_id.get(role)

                        if not agent_id or agent_id not in executors:
                            logger.warning(f"No executor for node {step} (role: {role})")
                            continue

                        executor = executors[agent_id]
                        agent_config = agent_configs[agent_id]

                        # Build context from previous agent outputs
                        context = "\n".join(
                            f"[{m.get('agent_name', 'Agent')}]: {m['content']}"
                            for m in state["messages"]
                            if m.get("type") == "agent_output"
                        )

                        # Set agent status to running
                        agent_obj = await db.execute(select(Agent).where(Agent.id == agent_id))
                        agent_record = agent_obj.scalar_one_or_none()
                        if agent_record:
                            agent_record.status = "running"
                            await db.commit()

                        # Execute agent
                        input_text = state["current_output"] or state["input"]
                        await self._broadcast_event("agent_start", {
                            "execution_id": execution_id,
                            "agent_id": agent_id,
                            "agent_name": agent_config["name"],
                        })

                        result = await executor.execute(input_text, context=context if context else None)

                        state["current_output"] = result["content"]
                        state["iteration"] += 1
                        total_tokens += result["tokens_used"]
                        total_cost += result["cost"]
                        final_output = result["content"]

                        state["messages"].append({
                            "type": "agent_output",
                            "agent_name": agent_config["name"],
                            "agent_id": agent_id,
                            "content": result["content"],
                        })

                        # Log the inter-agent message
                        await self._log_message(
                            db, execution_id, agent_id, None,
                            result["content"], "agent_output",
                            tokens=result["tokens_used"], cost=result["cost"]
                        )

                        # Broadcast to monitoring
                        await self._broadcast_event("agent_output", {
                            "execution_id": execution_id,
                            "agent_id": agent_id,
                            "agent_name": agent_config["name"],
                            "content": result["content"][:500],
                            "tokens_used": result["tokens_used"],
                            "cost": result["cost"],
                        })

                        # Set agent status back to idle
                        if agent_record:
                            agent_record.status = "idle"
                            await db.commit()

                # Update execution record
                result_exec = await db.execute(
                    select(WorkflowExecution).where(WorkflowExecution.id == execution_id)
                )
                execution = result_exec.scalar_one_or_none()
                if execution:
                    execution.status = "completed"
                    execution.result = {
                        "output": final_output,
                        "messages": [
                            {"agent": m.get("agent_name", ""), "content": m.get("content", "")}
                            for m in state["messages"] if m.get("type") == "agent_output"
                        ],
                    }
                    execution.total_tokens = total_tokens
                    execution.total_cost = total_cost
                    execution.completed_at = datetime.datetime.utcnow()
                    await db.commit()

                await self._broadcast_event("workflow_complete", {
                    "execution_id": execution_id,
                    "status": "completed",
                    "total_tokens": total_tokens,
                    "total_cost": total_cost,
                })

        except Exception as e:
            logger.error(f"Workflow execution error: {e}", exc_info=True)
            try:
                async with async_session() as db:
                    result = await db.execute(
                        select(WorkflowExecution).where(WorkflowExecution.id == execution_id)
                    )
                    execution = result.scalar_one_or_none()
                    if execution:
                        execution.status = "failed"
                        execution.result = {"error": str(e)}
                        execution.completed_at = datetime.datetime.utcnow()
                        await db.commit()
            except Exception:
                pass

            await self._broadcast_event("workflow_error", {
                "execution_id": execution_id,
                "error": str(e),
            })

    def _resolve_execution_order(self, nodes: list, edges: list) -> list[str]:
        """Resolve a linear execution order from the graph (topological sort)."""
        # Build adjacency list (skip feedback edges marked with conditions)
        graph: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}

        for edge in edges:
            # For the initial pass, prefer the "happy path" (non-revision edges)
            if edge.get("condition") == "revision_needed":
                continue
            graph[edge["source"]].append(edge["target"])
            in_degree[edge["target"]] = in_degree.get(edge["target"], 0) + 1

        # Topological sort (BFS - Kahn's algorithm)
        queue = [n for n, d in in_degree.items() if d == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    def _evaluate_condition(self, node: dict, current_output: str, edges: list) -> str:
        """Evaluate a condition node based on the current output."""
        # Simple keyword-based condition evaluation
        output_lower = current_output.lower()

        # Check for approval/rejection keywords
        approval_keywords = ["approved", "looks good", "well-structured", "accept", "satisfied", "complete"]
        rejection_keywords = ["revision", "revise", "improve", "needs work", "missing", "incomplete", "reject"]

        approval_score = sum(1 for kw in approval_keywords if kw in output_lower)
        rejection_score = sum(1 for kw in rejection_keywords if kw in output_lower)

        # Check for escalation keywords
        escalation_keywords = ["escalate", "complex", "specialist", "unable to resolve", "advanced"]
        escalation_score = sum(1 for kw in escalation_keywords if kw in output_lower)

        if escalation_score > 0:
            return "escalate"
        if rejection_score > approval_score:
            return "revision_needed"
        return "approved" if approval_score > 0 else "resolved"

    async def _log_message(
        self, db: AsyncSession, execution_id: str,
        from_agent_id: Optional[str], to_agent_id: Optional[str],
        content: str, message_type: str,
        tokens: int = 0, cost: float = 0.0,
    ):
        """Log a message to the database."""
        msg = Message(
            id=generate_uuid(),
            execution_id=execution_id,
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            content=content,
            message_type=message_type,
            tokens_used=tokens,
            cost=cost,
        )
        db.add(msg)
        await db.commit()

    async def _broadcast_event(self, event_type: str, data: dict):
        """Broadcast event to WebSocket clients."""
        try:
            from api.monitoring import broadcast_event
            await broadcast_event(event_type, data)
        except Exception as e:
            logger.debug(f"Broadcast error (non-critical): {e}")
