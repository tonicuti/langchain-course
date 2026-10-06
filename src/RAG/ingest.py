from dotenv import load_dotenv
from pathlib import Path

import os

load_dotenv(Path(__file__).parent / ".env")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter



if __name__ == "__main__":
    print(os.environ["PINECONE_API_KEY"])