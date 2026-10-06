from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# ===== TOOLS (LangChain Tools Decorate) ===== #
# Different 1: Without LangChain, we must MANUALLY trace the tool calls for LangSmith
@traceable(run_type = "tool")
def get_product_price(product_name: str) -> str:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price: Getting price for: {product_name}")
    price = {'Laptop': 999.99, 'Smartphone': 499.99, 'Headphones': 199.99}
    return f"The price of {product_name} is ${price.get(product_name, 'N/A')}"

@traceable(run_type = "tool")
def apply_discount(product_name: str, discount_code: str) -> str:
    """Apply a discount code to a product."""
    print(f"    >> Executing apply_discount: Applying discount for: {product_name} with code: {discount_code}")
    discount_codes = {'SAVE10': 0.10, 'SAVE20': 0.20}
    discount = discount_codes.get(discount_code, 0)
    price = {'Laptop': 999.99, 'Smartphone': 499.99, 'Headphones': 199.99}
    discounted_price = price.get(product_name, 0) * (1 - discount)
    return f"The discounted price of {product_name} is ${discounted_price:.2f}"

# Different 2: Without @tool, we must MANUALLY design the JSON schema for each function
# This is exactly what LangChain's @tool decorator does automatically from the function's type hints and docstring.
tools_for_llm = [
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
                        "description": "The name of the product to look up."
                    }
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount code to a product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The name of the product to discount."
                    },
                    "discount_code": {
                        "type": "string",
                        "description": "The discount code to apply, such as SAVE10 or SAVE20."
                    }
                },
                "required": ["product_name", "discount_code"]
            }
        }
    }
]


# Different 3:  Without LangChain, we must MANUALLY trace LLM calls for LangSmith
@traceable(name = "Ollama Chat", run_type = "llm")
def ollama_chat_traced(messages):
    return ollama.chat(model = MODEL, tools = tools_for_llm, messages = messages)

@traceable(name = "Traceable Agent Loop")
def run_agent(query: str):
    # Different 4: Without LangChain, we must MANUALLY create a dictionary of tool names to tool functions for calling them.
    tool_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }
    
    
    print(f"User Query: {query}")
    print("=" * 60)
    
    # This format messages works for the Ollama API.
    # Other providers may require different message formats.
    messages = [
        {
            "role": "system",
            "content": """
                You are a helpful shop assistant.
                You have access to a product catalog tool and a discount tool.
                
                REQUIREMENTS:
                1. NEVER GUESS the price of a product. If you don't know, use the get_product_price tool to get the REAL PRICE.
                2. Only call apply_discount AFTER you have obtained the REAL PRICE of the product using get_product_price.
                3. NEVER calculate the discounted price yourself. Always use the apply_discount tool to get the discounted price.
            """
        },
        {
            "role": "user", 
            "content": query
        },
    ]
    
    for it in range(1, MAX_ITERATIONS + 1):
        print(f"\n ==== Iteration {it}: ====")
        
        response = ollama_chat_traced(messages)
        ai_message = response.message
        
        tool_calls = ai_message.tool_calls
        
        if not tool_calls:
            print(f"    >> AI Response: {ai_message.content}")
            return ai_message.content
        
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        
        print(f"    >> AI is calling tool: {tool_name} with args: {tool_args}")
        
        tool_to_use = tool_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found.")
        
        # Different 5: Without LangChain, we must MANUALLY call the tool function with the arguments.
        observation = tool_to_use(**tool_args)
        
        print(f"    >> Tool Observation: {observation}")
        
        messages.append(ai_message)
        messages.append({
            "role": "tool",
            "content": str(observation),
        })
        
    print("ERROR: Maximum iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    user_query = "What is the price of a Laptop and can I get a discount with code SAVE10?"
    final_response = run_agent(user_query)
    print("=" * 60)
    print(f"Final Response: {final_response}")
        
