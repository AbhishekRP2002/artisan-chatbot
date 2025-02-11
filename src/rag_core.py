from langchain_sambanova import ChatSambaNovaCloud
from langchain_openai import AzureChatOpenAI
import os
from .utils import get_vs_as_retriever
from .prompts import BASE_SYSTEM_PROMPT
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatSambaNovaCloud(
    sambanova_api_key=os.environ.get("SAMBANOVA_API_KEY"),
    model="Meta-Llama-3.3-70B-Instruct",
    temperature=0.1,
    max_tokens=1024,
)


llm_azure = AzureChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    azure_deployment="gpt-4o-mini",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-07-01-preview",
    max_tokens=1024,
)

retriever = get_vs_as_retriever()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", BASE_SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=qa_chain)


def get_response(query: str, session_id: str):
    response = rag_chain.invoke({"input": query})
    logger.info(response)
    return response["answer"]
