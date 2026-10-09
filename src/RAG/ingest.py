from dotenv import load_dotenv
from pathlib import Path

import os

load_dotenv(Path(__file__).parent / ".env")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore


if __name__ == "__main__":
    data_file = Path(__file__).resolve().parents[2] / "data" / "mediumblog1.txt"
    loader = TextLoader(data_file, encoding = "utf-8")
    document = loader.load()
    
    print("Splitting ...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    
    print(f"Created {len(texts)} chunks.")
    
    print("Creating embeddings ...")
    embeddings = OllamaEmbeddings(model="embeddinggemma")
    
    print("Ingesting to Pinecone ...")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=index_name)
        
    print("Finished")
