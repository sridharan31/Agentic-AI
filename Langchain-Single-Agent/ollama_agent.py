import json
import os
from typing import Any, Sequence

import requests
from langchain_core.tools import BaseTool


def _ollama_tool_schema(tool: BaseTool) -> dict[str, Any]:
    schema = tool.args_schema.schema() if tool.args_schema else {
        "type": "object",
        "properties": {},
    }
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": schema.get("properties", {}),
    }
    if schema.get("required"):
        parameters["required"] = schema["required"]

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": parameters,
        },
    }


def _select_relevant_tools(query: str, tools: Sequence[BaseTool]) -> list[BaseTool]:
    normalized_query = query.casefold()
    weather_request = any(
        term in normalized_query
        for term in ("weather", "temperature", "forecast", "humidity")
    )
    search_request = any(
        term in normalized_query
        for term in (
            "search",
            "look up",
            "latest",
            "recent",
            "news",
            "breaking",
            "price",
            "website",
            "web",
            "research",
        )
    )

    relevant_tools = []
    for tool in tools:
        description = f"{tool.name} {tool.description or ''}".casefold()
        if "weather" in description or "temperature" in description:
            if weather_request:
                relevant_tools.append(tool)
        elif "search" in description:
            if search_request:
                relevant_tools.append(tool)
        else:
            relevant_tools.append(tool)
    return relevant_tools


def invoke_ollama_agent(
    query: str,
    tools: Sequence[BaseTool],
    model: str = "llama3.2:latest",
    max_tool_rounds: int = 5,
) -> str:
    if max_tool_rounds < 1:
        raise ValueError("max_tool_rounds must be at least 1.")

    if len({tool.name for tool in tools}) != len(tools):
        raise ValueError("Ollama tools must have unique names.")
    relevant_tools = _select_relevant_tools(query, tools)
    tools_by_name = {tool.name: tool for tool in relevant_tools}

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use the available tools when they "
                "are relevant, then answer the user directly using the tool results. "
                "When you use a tool, treat its output as the only source for factual "
                "claims. If it does not contain enough information, say so; do not "
                "invent details."
            ),
        },
        {"role": "user", "content": query},
    ]
    ollama_tools = [_ollama_tool_schema(tool) for tool in relevant_tools]
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    tool_rounds = 0

    while True:
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "tools": ollama_tools,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        assistant_message = response.json()["message"]
        tool_calls = assistant_message.get("tool_calls", [])

        if not tool_calls:
            answer = assistant_message.get("content", "").strip()
            if not answer:
                raise RuntimeError("Ollama returned an empty answer.")
            return answer

        if tool_rounds >= max_tool_rounds:
            raise RuntimeError(
                f"Ollama exceeded the limit of {max_tool_rounds} tool-call rounds."
            )

        messages.append(assistant_message)
        for tool_call in tool_calls:
            function = tool_call["function"]
            tool_name = function["name"]
            tool = tools_by_name.get(tool_name)
            if tool is None:
                raise ValueError(f"Ollama requested an unavailable tool: {tool_name}")

            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"Ollama returned invalid arguments for tool {tool_name}."
                )

            result = tool.invoke(arguments)
            content = (
                result
                if isinstance(result, str)
                else json.dumps(result, ensure_ascii=False, default=str)
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": content,
                }
            )

        tool_rounds += 1
