import os

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage 
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

load_dotenv()

print("Initializing chat model ...")

embedding = OllamaEmbeddings(model="embeddinggemma")

llm = init_chat_model(
    model="qwen3:1.7b",
    model_provider="ollama",
)

vectorstore = PineconeVectorStore(
    embedding=embedding,
    index_name=os.environ.get("PINECONE_INDEX_NAME")
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Answer the question based on the following context:

{context}

Question: {question}

Provide a detailed answer:""",
        )
    ]
)

def format_prompt(docs):
    """Format the retrieved documents into a single string for the prompt."""
    return "\n\n".join([doc.page_content for doc in docs])


# IMPLEMENTATION 1: Without LCEL (Simple function-based approach)
def retrieval_chain_without_lcel(query: str):
    """ 
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates the respons.
    
    Limitation:
    - Manual step-by-step execution.
    - No built-in streaming support
    - No async support without additional code.
    - Harder to compose with other chains. 
    - More verbose and error-prone.
    """
    
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)
    
    # Step 2: Format documents into context string
    context = format_prompt(docs)
    
    # Step 3: Format the prompt with context and question
    prompt = prompt_template.format_prompt(context=context, question=query)
    
    # Step 4: Invoke the LLM with the formatted prompt
    result = llm.invoke(prompt.to_messages())
    
    return result

# IMPLEMENTATION 2: With LCEL (LangChain Execution Layer) - BETTER APPROACH
def create_retrieval_chain_with_lcel():
    """ 
    Create a retrieval chain using LCEL.
    Return a chain that can be invoke with {"question": "..."}
    
    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator {|}
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system.
    - Less code: More concise and readable.
    - Reuseable: Chain can be saved, shared, and composed with other chains.
    - Better debugging: LangChain provides better observability tools.
    """
    
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_prompt
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain

if __name__ == "__main__":
    print("Starting RAG ...")
    
    query = "What is the Pinecone vector database?"
    
    # ======================================= #
    # Option 0: Raw invocation without RAG
    # ======================================= #
    print("\n" + "="*70)
    print("\nOption 0: Raw invocation without RAG")
    print("="*70 + "\n")
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("Raw result:")
    print(result_raw.content)
    
    
    # ======================================= #
    # Option 1: Retrieval chain without LCEL
    # ======================================= #
    print("\n" + "="*70)
    print("\nOption 1: Retrieval chain without LCEL")
    print("="*70 + "\n")
    result_retrieval = retrieval_chain_without_lcel(query)
    print("Retrieval result:")
    print(result_retrieval.content)

    # ======================================= #
    # Option 2: Retrieval chain with LCEL
    # ======================================= #
    print("\n" + "="*70)
    print("\nOption 2: Retrieval chain with LCEL")
    print("="*70 + "\n")
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("Retrieval result with LCEL:")
    print(result_with_lcel)
