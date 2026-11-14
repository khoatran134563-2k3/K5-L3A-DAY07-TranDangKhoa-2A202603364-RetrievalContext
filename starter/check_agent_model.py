"""Check OpenAI model access and a real search tool call."""

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


DAY6_STARTER = Path(
    r"D:\VIN Lab\K5-L3A-DAY06-TranDangKhoa-2A202603364-DataIngestion\starter"
)

load_dotenv(DAY6_STARTER / ".env", override=False)

MODEL = "gpt-4.1-mini"


def main():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENAI_API_KEY is missing.")

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Find evidence in the document corpus before answering. "
                    "Use the search tool with the user's question."
                ),
            },
            {
                "role": "user",
                "content": (
                    "What institutional condition is included "
                    "in the report's analog complements?"
                ),
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the indexed report for evidence.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query.",
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": "search"},
        },
        "parallel_tool_calls": False,
        "max_completion_tokens": 200,
        "temperature": 0,
        "store": False,
    }

    request = Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
    except HTTPError as error:
        raise SystemExit(
            f"OpenAI HTTP {error.code}. "
            "Check model access, key permissions and quota."
        ) from None
    except (URLError, TimeoutError, ConnectionError):
        raise SystemExit("Could not connect to OpenAI.") from None

    choice = result["choices"][0]
    calls = choice["message"].get("tool_calls", [])

    if choice.get("finish_reason") not in {"tool_calls", "stop"} or len(calls) != 1:
        print("Returned model:", result.get("model"))
        print("Finish reason:", choice.get("finish_reason"))
        print("Tool call count:", len(calls))
        print(
            "Message:",
            json.dumps(
                choice.get("message"),
                ensure_ascii=False,
                indent=2,
            ),
        )
        print("Usage:", result.get("usage"))
        raise SystemExit("Tool-call check failed; see details above.")

    call = calls[0]
    arguments = json.loads(call["function"]["arguments"])

    if (
        call["function"]["name"] != "search"
        or not isinstance(arguments.get("query"), str)
        or not arguments["query"].strip()
    ):
        raise SystemExit("Invalid search tool call.")

    print(f"Requested model: {MODEL}")
    print(f"Returned model: {result.get('model')}")
    print(f"Tool: {call['function']['name']}")
    print(f"Arguments: {json.dumps(arguments, ensure_ascii=False)}")
    print(f"Usage: {result.get('usage', {})}")
    print("Model access and tool calling OK.")


if __name__ == "__main__":
    main()