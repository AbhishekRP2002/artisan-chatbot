from langchain_sambanova import ChatSambaNovaCloud
from langchain_openai import AzureChatOpenAI
import os
import logging
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from .utils import get_vs_as_retriever
from .prompts import BASE_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatbotRAG:
    def __init__(self):
        self.llm = self.initialize_llms()
        self.retriever = self.initialize_retriever()
        self.rag_chain = self.initialize_rag_chain()
        self.store = {}

    def initialize_llms(self):
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

        return llm.with_fallbacks([llm_azure])

    def initialize_retriever(self):
        retriever = get_vs_as_retriever()
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Given a chat history and the latest user question "
                    "which might reference context in the chat history, "
                    "formulate a standalone question which can be understood "
                    "without the chat history. Do NOT answer the question, "
                    "just reformulate it if needed and otherwise return it as is.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        return create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

    def initialize_rag_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", BASE_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history", n_messages=10),
                ("human", "{input}"),
            ]
        )
        qa_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(
            retriever=self.retriever, combine_docs_chain=qa_chain
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def get_response(self, query: str, session_id: str):
        conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
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


chatbot = ChatbotRAG()
