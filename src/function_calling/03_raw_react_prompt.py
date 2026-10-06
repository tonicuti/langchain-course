#CHANGE 1: Add re + inspect - We'll parse tool calls from the raw text instead of structured JSON.
import re
import inspect

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
def get_product_price(product_name: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price for: {product_name}")
    
    price = {'Laptop': 999.99, 'Smartphone': 499.99, 'Headphones': 199.99}
    
    return price.get(product_name, 0.0)

@traceable(run_type = "tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a product."""
    print(f"    >> Executing apply_discount for: {price} with tier: {discount_tier}")
    price = float(price)
    discount_tier = {'bronze': 0.10, 'silver': 0.20, 'gold': 0.30}
    discount = discount_tier.get(discount_tier, 0)
    discounted_price = round(price * (1 - discount), 2)
    
    return discounted_price


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

#CHANGE 2: Delete the JSON schemas. Tools now live inside the prompt as plain text
# We derive descriptions from the functions themselves using the inspect module.

def get_tool_descriptions(tools_dict):
    descriptions = []
    
    for tool_name, tool_func in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (e.g., @traceable, @tool, etc.) to get the original function.
        original_func = getattr(tool_func, "__wrapped__", tool_func)
        signature = inspect.signature(original_func)
        docstring = inspect.getdoc(original_func) or "No description available."
        descriptions.append(f"{tool_name}{signature} - {docstring}")
        
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES - you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount after you have received a price from get_product_price. Pass the exact price returned by get_product_price - do NOT calculate the discounted price yourself.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use - do nOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {{question}}
Thought:
"""

#CHANGE 4: Drop tools = from ollama.chat(). The LLM has no idea it's an agent - 
# all agency comes from the prompt above and our regex parsing below.

@traceable(name = "Ollama Chat", run_type = "llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model = model, messages = messages, options = options)

# ---- AGENT LOOP ---- #
@traceable(name = "Traceable Agent Loop")
def run_agent(question: str):
    print(f"User question: {question}")
    print("=" * 60)
    
    # CHANGE 5: One prompt string replaces the system/ user message split
    prompt = react_prompt.format(question=question)
    scratchpad = ""     #Containing the history of thoughts, actions, and observations of the agent.
    
    for it in range(1, MAX_ITERATIONS + 1):
        print(f"\n ==== Iteration {it}: ====")
        
        full_prompt = prompt + scratchpad
        
        response = ollama_chat_traced(
            model = MODEL,
            messages = [{"role": "user", "content": full_prompt}],
            options = {"stop": ["\nObservations"], "temperature": 0}
        )
        output = response.message.content
        print(f"   >> LLM Output:\n{output}")
        
        print(f"    [Parsing] Looking for the final answer in LLM Output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"    [Parsed] Final Answer Found: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer
        
    
        # CHANGE 6: Parse the tool calls from the raw text with regex - fragile if LLM doesn't follow the format.
        print(f"    [Parsing] Looking for Action and Action Input in LLM Output...")
        
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)
        
        if not action_match or not action_input_match:
            print("    [Parsing] ERROR: Could not parse Action or Action Input from LLM output.")
            break
        
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        
        print(f"    [Tool Selected]: {tool_name} with args: {tool_input_raw}")
        
        # Split comma-separated args; strip key = prefix if LLM outputs key = value format.
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [s.split("=", 1)[-1].strip().strip('"\'') for s in raw_args]

        print(f"    [Tool Executing] {tool_name}({args})...")
        
        if(tool_name not in tools):
            observation = f"Error: Tool {tool_name} not found. Available tools: list({', '.join(tools.keys())})"
        else:
            observation = tools[tool_name](*args)
            
        print(f"    [Tool Observation]: {observation}")
        
        # CHANGE 7: History is one growing string re-sent every iteration (Replace messages.append()).
        scratchpad += f"{output}\nObservation: {observation}\nThought:"
        
    print("ERROR: Maximum iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    user_question = "What is the price of a Laptop with a discount tier silver?"
    final_response = run_agent(user_question)
        
