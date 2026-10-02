from dotenv import load_dotenv

import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

load_dotenv()

@tool
def search(query: str) -> str:
    """_summary_

    Args:
        query (str): _description_

    Returns:
        str: _description_
    """

    print(f"Searching for: {query}")
    return f"Results for: {query}"

llm = ChatOllama(
    model="qwen2.5:1.5b",
    base_url=os.environ.get("OLLAMA_BASE_URL"),
    temperature=0,
)

tools = [search]

agent = create_agent(
    model=llm,
    tools=tools,
)

def main():
    result = agent.invoke(
        {"messages": HumanMessage(content="Search for the latest news about Elon Musk")},
    )
    print(result)
    
if __name__ == "__main__":
    main()