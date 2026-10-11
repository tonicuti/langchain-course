import asyncio
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

try:
    from .logger import (
        Colors,
        log_info,
        log_success,
        log_error,
        log_warning,
        log_header,
    )
except ImportError:
    from logger import (
        Colors,
        log_info,
        log_success,
        log_error,
        log_warning,
        log_header,
    )

load_dotenv() 


embeddings = OllamaEmbeddings(model="embeddinggemma:latest")


vector_store = PineconeVectorStore(
    index_name=os.getenv("PINECONE_NAME_INDEX"),
    embedding=embeddings,
)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth = 5, max_breadth = 20, max_pages = 1000)
tavily_crawl = TavilyCrawl()

def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    """Chunk a list of URLs into smaller lists of a specified size."""
    return [urls[i:i + chunk_size] for i in range(0, len(urls), chunk_size)]
    
async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs using TavilyExtract."""
    
    try:
        log_info(
            f"  [TAVILY EXTRACT] Extracting batch {batch_num} with {len(urls)} URLs.",
            Colors.BLUE
        )
        
        docs = await tavily_extract.ainvoke(input = {"urls": urls})
        
        log_success(
            f"  [TAVILY EXTRACT] Successfully extracted {len(docs.get('results', []))} documents from batch {batch_num}."
        )
        
        return docs
    except Exception as e:
        log_error(
            f"  [TAVILY EXTRACT] Error occurred while extracting batch {batch_num}: {e}"
        )
        return []
    
async def async_extract(url_batches: List[List[str]]) -> List[Document]:
    """Asynchronously extract documents from multiple batches of URLs."""
    
    log_header(
        "DOCUMENT EXTRACTION PHASE"
    )
    
    log_info(
        f"  [TAVILY EXTRACT] Starting concurrent extraction for {len(url_batches)} batches of URLs.",
        Colors.DARKCYAN
    )
    
    tasks = [
        extract_batch(batch, batch_num + 1)
        for batch_num, batch in enumerate(url_batches)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Normalize each batch response, then build Documents from valid pages.
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            failed_batches += 1
            log_error(
                f"  [TAVILY EXTRACT] A batch extraction failed with error: {result}"
            )
            continue

        # TavilyExtract may return a response dict (with "results") or a list
        # of page records, depending on the integration/tool response shape.
        pages = result.get("results", []) if isinstance(result, dict) else result
        if not isinstance(pages, list):
            failed_batches += 1
            log_error(
                f"  [TAVILY EXTRACT] Batch returned an unexpected response: {type(result).__name__}."
            )
            continue

        for extract_page in pages:
            if not isinstance(extract_page, dict):
                continue

            raw_content = extract_page.get("raw_content")
            if not isinstance(raw_content, str) or not raw_content.strip():
                continue

            all_pages.append(
                Document(
                    page_content=raw_content,
                    metadata={"source": extract_page.get("url", "")},
                )
            )
                
    log_success(
        f"  [TAVILY EXTRACT] Successfully extracted a total of {len(all_pages)} documents from {len(url_batches) - failed_batches} successful batches.",
    )
    
    if failed_batches > 0:
        log_warning(
            f"  [TAVILY EXTRACT] {failed_batches} batch(es) failed during extraction."
        )
        
    return all_pages


async def main():
    """Main async function to orchestrate the entire processing pipeline."""
    log_header("DOCUMENTATION INGESTION PIPELINE")
    
    log_info(
        "  [TAVILY CRAWL] Starting the Tavily crawl from https://python.langchain.com/",
        Colors.PURPLE
    )
    
    site_map = tavily_map.invoke("https://python.langchain.com/")
    log_success(
        f"  [TAVILY MAP] Successfully mapped {len(site_map)} URLs."
    )
    
    #Splits URLs into batches of 20
    url_batches = chunk_urls(list(site_map["results"]), chunk_size=20)
    
    log_info(
        f"  [TAVILY CRAWL] Split {len(site_map["results"])} URLs into {len(url_batches)} batches of URLs.",
        Colors.BLUE
    )
    
    # Extract documents from URLs
    all_docs = await async_extract(url_batches)
    
    # Split documents into smaller chunks for better embedding and storage
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"  [CHUNKING] Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap.",
        Colors.YELLOW
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200
    )
    
    splitted_docs = text_splitter.split_documents(all_docs)
    
    log_success(
        f"  [CHUNKING] Successfully split documents into {len(splitted_docs)} chunks."
    )


if __name__ == "__main__":
    asyncio.run(main())







