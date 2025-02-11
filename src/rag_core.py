from langchain_sambanova import ChatSambaNovaCloud
from langchain_openai import AzureChatOpenAI
import os
from .utils import get_vs_as_retriever
from .prompts import BASE_SYSTEM_PROMPT
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import logging
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage  # noqa
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", BASE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", n_messages=10),
        ("human", "{input}"),
    ]
)
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever, combine_docs_chain=qa_chain
)
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_response(query: str, session_id: str):
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    response = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    )
    logger.info(response)
    return response["answer"]
