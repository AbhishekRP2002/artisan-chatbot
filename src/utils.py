"""
extract documents and store them in a vector db as a collection
"""

from .web_scrape import scrape_main
from langchain_openai import AzureOpenAIEmbeddings
from langchain_milvus import Milvus
import os
from dotenv import load_dotenv
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from concurrent.futures import ProcessPoolExecutor
import asyncio
import time
from itertools import chain
from rich.pretty import pprint

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


embedding_model = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
)


def get_milvus_vector_store():
    return Milvus(
        embedding_function=embedding_model,
        collection_name="test_collection",
        connection_args={"uri": os.getenv("MILVUS_URI")},
        auto_id=True,
        drop_old=False,
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 8, "efConstruction": 64},
        },
    )


def get_vs_as_retriever():
    return get_milvus_vector_store().as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )


URLS_TO_SCRAPE = [
    "https://www.artisan.co",
    "https://www.artisan.co/about",
    "https://www.artisan.co/sales-ai",
    "https://www.artisan.co/ai-sales-agent",
    "https://www.artisan.co/products/linkedin-outreach",
    "https://www.artisan.co/products/email-warmup",
    "https://www.artisan.co/products/sales-automation",
    "https://www.artisan.co/products/email-personalization",
    "https://www.artisan.co/features/email-deliverability",
    "https://help.artisan.co/articles/7415399613-can-i-schedule-when-emails-go-out",
    "https://help.artisan.co/articles/5365244006-what-is-email-warmup",
    "https://help.artisan.co/articles/8442274387-ava-is-sending-strange-messages-from-my-email",
    "https://help.artisan.co/articles/1195138264-is-there-a-limit-to-the-amount-of-leads-i-can-have-in-my-csv-file",
    "https://help.artisan.co/articles/5617649387-help-i-can-t-turn-on-my-campaign",
    "https://help.artisan.co/articles/1048710797-how-does-website-visitor-identification-work",
    "https://help.artisan.co/articles/3886727025-generate-sample-email",
    "https://help.artisan.co/articles/6218358204-running-ava-on-copilot-vs-autopilot",
    "https://help.artisan.co/articles/9265896700-adding-delegates-and-team-members",
    "https://help.artisan.co/articles/2734968853-how-to-create-a-campaign",
    "https://help.artisan.co/articles/7633990298-how-to-integrate-artisan-with-your-crm",
    "https://help.artisan.co/articles/6092562650-how-do-i-upload-a-csv-file-of-my-own-leads",
    "https://help.artisan.co/articles/4356675492-how-do-i-add-variables-to-my-email",
    "https://help.artisan.co/articles/3551943296-how-do-i-request-a-script-tag-for-my-watchtower-campaign",
    "https://help.artisan.co/articles/9602711709-how-do-i-integrate-ava-with-slack",
    "https://www.artisan.co/pricing",
]


def chunk_parent_document(document: Document) -> list[Document]:
    semantic_text_splitter = SemanticChunker(
        embeddings=embedding_model, min_chunk_size=100
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        separators=[
            "\n#{1,6} ",
            "\n\\*\\*\\*+\n",
            "\n---+\n",
            "\n___+\n",
            "\n\n",
            "\n",
            ".",
            "?",
            "!",
        ],
    )
    chunked_documents = semantic_text_splitter.split_documents([document])
    chunked_documents = [
        chunked_doc_item
        for chunked_doc_item in chunked_documents
        if len(chunked_doc_item.page_content) > 0 or chunked_doc_item.page_content
    ]
    final_chunked_documents = []

    for idx, chunked_doc in enumerate(chunked_documents):
        pprint(chunked_doc)
        if len(chunked_doc.page_content) > 5000:
            sub_chunked_documents = text_splitter.split_documents([chunked_doc])
            final_chunked_documents.extend(sub_chunked_documents)
        else:
            final_chunked_documents.append(chunked_doc)
    return final_chunked_documents


async def ingest_urls(chunk_executor: ProcessPoolExecutor, milvus_vector_store: Milvus):
    lc_documents = await scrape_main(URLS_TO_SCRAPE)
    start_time = time.time()
    chunked_documents = list(chunk_executor.map(chunk_parent_document, lc_documents))
    chunked_documents = list(chain.from_iterable(chunked_documents))
    end_time = time.time()
    logger.info(f"Time taken to chunk documents: {end_time - start_time} seconds")

    start_time = time.time()
    if chunked_documents:
        _ = milvus_vector_store.add_documents(chunked_documents, batch_size=50)
    logger.info(
        f"Time taken to ingest documents into Milvus: {time.time() - start_time} seconds"
    )


if __name__ == "__main__":
    chunk_executor = ProcessPoolExecutor()
    milvus_vector_store = get_milvus_vector_store()
    asyncio.run(ingest_urls(chunk_executor, milvus_vector_store))
