from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# ===== TOOLS (LangChain Tools Decorate) ===== #
# @tool is automatically generating the JSON schema for each function from the function's type hints and docstring.

@tool
def get_product_price(product_name: str) -> str:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price: Getting price for: {product_name}")
    price = {'Laptop': 999.99, 'Smartphone': 499.99, 'Headphones': 199.99}
    return f"The price of {product_name} is ${price.get(product_name, 'N/A')}"

@tool
def apply_discount(product_name: str, discount_code: str) -> str:
    """Apply a discount code to a product."""
    print(f"    >> Executing apply_discount: Applying discount for: {product_name} with code: {discount_code}")
    discount_codes = {'SAVE10': 0.10, 'SAVE20': 0.20}
    discount = discount_codes.get(discount_code, 0)
    price = {'Laptop': 999.99, 'Smartphone': 499.99, 'Headphones': 199.99}
    discounted_price = price.get(product_name, 0) * (1 - discount)
    return f"The discounted price of {product_name} is ${discounted_price:.2f}"

@traceable(name = "Traceable Agent Loop")
def run_agent(query: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    
    llm = init_chat_model(f"ollama: {MODEL}", temperature=0)    # Enable provider switching model in 1 line.
    llm_with_tools = llm.bind_tools(tools)      # Bind the tools to the model so it can call them during the conversation.
    
    print(f"User Query: {query}")
    print("=" * 60)
    
    messages = [
        SystemMessage(
            content = """
                You are a helpful shop assistant.
                You have access to a product catalog tool and a discount tool.
                
                REQUIREMENTS:
                1. NEVER GUESS the price of a product. If you don't know, use the get_product_price tool to get the REAL PRICE.
                2. Only call apply_discount AFTER you have obtained the REAL PRICE of the product using get_product_price.
                3. NEVER calculate the discounted price yourself. Always use the apply_discount tool to get the discounted price.
            """
        ),
        HumanMessage(content=query),
    ]
    
    for it in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {it}:")
        
        # A bound LangChain model is a Runnable; call it with .invoke().
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls      # Return structured data about the tools the model wants to call instead of raw JSON.
        
        if not tool_calls:
            print(f"    >> AI Response: {ai_message.content}")
            return ai_message.content
        
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        
        print(f"    >> AI is calling tool: {tool_name} with args: {tool_args}")
        
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found.")
        
        observation = tool_to_use.invoke(tool_args)
        
        print(f"    >> Tool Observation: {observation}")
        
        messages.append(ai_message)
        messages.append(ToolMessage(
            content=str(observation),
            name=tool_name,
            tool_call_id=tool_call_id
        ))
        
    print("ERROR: Maximum iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    user_query = "What is the price of a Laptop and can I get a discount with code SAVE10?"
    final_response = run_agent(user_query)
    print("=" * 60)
    print(f"Final Response: {final_response}")
        
