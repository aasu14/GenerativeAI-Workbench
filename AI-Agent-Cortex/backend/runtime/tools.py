"""Tool registry for AI agents. Each tool is a callable that agents can invoke."""
import json
import logging
import math
import re
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)


async def web_search(query: str) -> str:
    """Search the web for information using DuckDuckGo."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_redirect": 1},
            )
            data = response.json()

            results = []
            if data.get("AbstractText"):
                results.append(f"Summary: {data['AbstractText']}")
            if data.get("AbstractSource"):
                results.append(f"Source: {data['AbstractSource']}")

            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append(f"- {topic['Text']}")

            if results:
                return "\n".join(results)
            return f"Search completed for '{query}'. No detailed results found. Try rephrasing."
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"Search error: {str(e)}"


async def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # Only allow safe math operations
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("_")
        }
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})

        # Remove anything that isn't a number, operator, or allowed function
        clean = re.sub(r'[^0-9+\-*/().,%\s a-zA-Z_]', '', expression)
        result = eval(clean, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


async def text_analysis(text: str) -> str:
    """Analyze text for basic metrics: word count, sentence count, reading level."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]

    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
    avg_sentence_length = word_count / sentence_count

    return json.dumps({
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_length, 1),
        "avg_sentence_length": round(avg_sentence_length, 1),
        "estimated_reading_time_minutes": round(word_count / 200, 1),
    })


async def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time."""
    return f"Current time (UTC): {datetime.utcnow().isoformat()}"


async def code_executor(code: str) -> str:
    """Execute simple Python expressions (sandboxed - no imports, no IO)."""
    try:
        # Very restricted execution environment
        result = eval(code, {"__builtins__": {}}, {  # noqa: S307
            "len": len, "str": str, "int": int, "float": float,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "range": range, "sorted": sorted, "reversed": reversed,
            "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
            "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
            "True": True, "False": False, "None": None,
        })
        return f"Output: {result}"
    except Exception as e:
        return f"Execution error: {str(e)}"


# Tool registry mapping
TOOL_REGISTRY: dict[str, dict] = {
    "web_search": {
        "function": web_search,
        "name": "web_search",
        "description": "Search the web for information on any topic",
        "parameters": {"query": "The search query string"},
    },
    "calculator": {
        "function": calculator,
        "name": "calculator",
        "description": "Evaluate mathematical expressions",
        "parameters": {"expression": "The math expression to evaluate"},
    },
    "text_analysis": {
        "function": text_analysis,
        "name": "text_analysis",
        "description": "Analyze text for word count, reading level, and other metrics",
        "parameters": {"text": "The text to analyze"},
    },
    "current_time": {
        "function": get_current_time,
        "name": "current_time",
        "description": "Get the current date and time",
        "parameters": {"timezone": "Timezone (default UTC)"},
    },
    "code_executor": {
        "function": code_executor,
        "name": "code_executor",
        "description": "Execute simple Python expressions",
        "parameters": {"code": "Python expression to evaluate"},
    },
}


def get_tools_for_agent(tool_names: list[str]) -> list[dict]:
    """Get tool definitions for a specific agent based on configured tool names."""
    tools = []
    for name in tool_names:
        if name in TOOL_REGISTRY:
            tool = TOOL_REGISTRY[name]
            tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            })
    return tools


async def execute_tool(tool_name: str, **kwargs) -> str:
    """Execute a tool by name with given arguments."""
    if tool_name not in TOOL_REGISTRY:
        return f"Unknown tool: {tool_name}"
    tool_func = TOOL_REGISTRY[tool_name]["function"]
    return await tool_func(**kwargs)
