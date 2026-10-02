"""A small tool-calling agent loop using Ollama directly, without LangChain."""

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def get_product_price(product_name: str) -> str:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price: Getting price for: {product_name}")
    price = {'Laptop': 999.99, 'Smartphone': 499.99, 'Headphones': 199.99}
    return f"The price of {product_name} is ${price.get(product_name, 'N/A')}"


def apply_discount(product_name: str, discount_code: str) -> str:
    """Apply a discount code to a product."""
    print(f"    >> Executing apply_discount: Applying discount for: {product_name} with code: {discount_code}")
    discount_codes = {'SAVE10': 0.10, 'SAVE20': 0.20}
    discount = discount_codes.get(discount_code, 0)
    price = {'Laptop': 999.99, 'Smartphone': 499.99, 'Headphones': 199.99}
    discounted_price = price.get(product_name, 0) * (1 - discount)
    return f"The discounted price of {product_name} is ${discounted_price:.2f}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The product name, such as Laptop.",
                    }
                },
                "required": ["product_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount code to a product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "discount_code": {"type": "string"},
                },
                "required": ["product_name", "discount_code"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

SYSTEM_PROMPT = """
You are a helpful shop assistant.
You have access to a product catalog tool and a discount tool.

Requirements:
1. Never guess a product price. Use get_product_price first.
2. Call apply_discount only after obtaining the real product price.
3. Never calculate a discounted price yourself. Use apply_discount.
""".strip()


def call_ollama(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Send one chat request to Ollama and return its decoded JSON response."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
    }
    request = Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {error.code}: {body}") from error
    except URLError as error:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_BASE_URL}: {error.reason}"
        ) from error


def run_agent(query: str) -> str | None:
    """Run the manual model/tool loop until the model returns a final answer."""
    print(f"User Query: {query}")
    print("=" * 60)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {iteration}:")
        response = call_ollama(messages)
        assistant_message = response.get("message", {})
        tool_calls = assistant_message.get("tool_calls", [])

        messages.append(assistant_message)

        if not tool_calls:
            answer = assistant_message.get("content", "")
            print(f"    >> AI Response: {answer}")
            return answer

        tool_call = tool_calls[0]
        function = tool_call.get("function", {})
        tool_name = function.get("name")
        tool_args = function.get("arguments", {})

        if isinstance(tool_args, str):
            tool_args = json.loads(tool_args)

        tool_to_use = TOOL_FUNCTIONS.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name!r} not found.")

        print(f"    >> AI is calling tool: {tool_name} with args: {tool_args}")
        observation = tool_to_use(**tool_args)
        print(f"    >> Tool Observation: {observation}")

        messages.append(
            {
                "role": "tool",
                "tool_name": tool_name,
                "content": observation,
            }
        )

    print("ERROR: Maximum iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    user_query = "What is the price of a Laptop and can I get a discount with code SAVE10?"
    final_response = run_agent(user_query)
    print("=" * 60)
    print(f"Final Response: {final_response}")
