from langchain_community.document_loaders.firecrawl import FireCrawlLoader
import os
from dotenv import load_dotenv
import asyncio
from rich.pretty import pprint  # noqa
from typing import List
from langchain_core.documents import Document
import re
import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import logging


load_dotenv()

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def clean_markdown(document: Document) -> Document:
    raw_content = document.page_content
    metadata = {
        "url": document.metadata.get("og:url", document.metadata.get("ogUrl", None)),
        "title": document.metadata.get(
            "og:title", document.metadata.get("ogTitle", None)
        ),
        "description": document.metadata.get(
            "og:description", document.metadata.get("ogDescription", None)
        ),
    }

    try:
        cleaned_content = re.sub(r"\!\[.*?\]\(.*?\)", "", raw_content)
        cleaned_content = re.sub(r"![.*?](.*?)", "", cleaned_content)
        cleaned_content = re.sub(r"\[.*?\]\(.*?\)", "", cleaned_content)
        cleaned_content = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned_content)
        cleaned_content = re.sub(r"\n\n\n+", "\n\n", cleaned_content)
        cleaned_content = re.sub(r"([^a-zA-Z0-9\s])\1{3,}", r"\1\1", cleaned_content)
        cleaned_content = re.sub(r"[\U0001F300-\U0001F9FF]+\n\n", "", cleaned_content)
        cleaned_content = re.sub(r"\n\n[/#]\n\n", "\n\n", cleaned_content)
        cleaned_content = cleaned_content.strip()
    except Exception as e:
        logger.error(f"Error cleaning markdown: {e}")
        raise e

    document.page_content = cleaned_content
    document.metadata = metadata

    return document


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=15),
    retry=retry_if_exception_type(
        (Exception, asyncio.TimeoutError, aiohttp.ClientError)
    ),
    retry_error_callback=lambda retry_state: None,
)
async def scrape_website(url: str):
    logger.info(f"Scraping url : {url}")
    try:
        lc_loader = FireCrawlLoader(
            url=url,
            api_key=FIRECRAWL_API_KEY,
            mode="scrape",
            params={
                "formats": ["markdown"],
                "onlyMainContent": True,
                "removeBase64Images": True,
                "skipTlsVerification": True,
            },
        )
        lc_doc = await lc_loader.aload()
        cleaned_lc_doc = clean_markdown(lc_doc[0])
        return cleaned_lc_doc
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        raise e


async def scrape_main(urls: List[str]):
    tasks = [scrape_website(url) for url in urls]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    return [
        response
        for response in responses
        if response is not None or isinstance(response, Exception)
    ]


if __name__ == "__main__":
    urls = ["https://www.artisan.co", "https://www.artisan.co/about"]
    responses = asyncio.run(scrape_main(urls))
    pprint(responses)
