import chainlit as cl
import httpx
import json  # noqa
from typing import Dict
from dotenv import load_dotenv
import uuid
import logging
from langchain_sambanova import ChatSambaNovaCloud
from langchain_openai import AzureChatOpenAI
import os
import sys
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from chainlit.input_widget import TextInput
from typing import cast, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import get_vs_as_retriever
from src.prompts import BASE_SYSTEM_PROMPT
from src.rag_core import ChatbotRAG

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

API_ENDPOINT = "https://artisan-chatbot.onrender.com/chat"


class ChatAPI:
    def __init__(self):
        self.client = httpx.AsyncClient(
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )

    async def send_message(self, message: str, session_id: str) -> Dict:
        try:
            response = await self.client.post(
                API_ENDPOINT, json={"message": message, "session_id": session_id}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise Exception(f"HTTP error occurred: {str(e)}")
        except Exception as e:
            raise Exception(f"Error sending message: {str(e)}")


chat_api = ChatAPI()


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    return default_user


@cl.on_chat_start
async def on_chat_start():
    max_attempts = 3
    attempt = 0
    company_name = None

    while attempt < max_attempts:
        try:
            company_domain_msg = cl.AskUserMessage(
                content="Please enter the company domain you want to chat about (e.g., example.com):",
                timeout=300,
                raise_on_timeout=True,
            )
            response = await company_domain_msg.send()
            company_name = response["output"]
            if company_name and company_name.strip():
                cl.user_session.set("company_domain", company_name)
                print(f"Company domain set to: {company_name}")
                break
        except Exception:
            attempt += 1
            if attempt < max_attempts:
                await cl.Message(
                    content=f"No company domain was provided. You have {max_attempts - attempt} attempt(s) left to enter a valid company domain."
                ).send()
            else:
                await cl.Message(
                    content="No company domain was provided after 3 attempts. Please refresh and try again, or contact support if the issue persists."
                ).send()
                return

    if not company_name:
        return

    chatbot_obj = ChatbotRAG()
    streaming_llm = ChatSambaNovaCloud(
        sambanova_api_key=os.environ.get("SAMBANOVA_API_KEY"),
        model="Meta-Llama-3.3-70B-Instruct",
        temperature=0.1,
        max_tokens=1024,
        streaming=True,
    )
    streaming_llm_azure = AzureChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        azure_deployment="gpt-4o-mini",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-07-01-preview",
        max_tokens=1024,
        streaming=True,
    )
    final_model = streaming_llm.with_fallbacks([streaming_llm_azure])
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
    company_system_prompt = f"{BASE_SYSTEM_PROMPT}\nYou are an assistant specialized in providing information about {company_name}. Only answer questions related to this company."
    history_aware_retriever = create_history_aware_retriever(
        streaming_llm, retriever, contextualize_q_prompt
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", company_system_prompt),
            MessagesPlaceholder("chat_history", n_messages=10),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(final_model, prompt)
    rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=qa_chain
    )
    conversational_rag_chain_runnable = RunnableWithMessageHistory(
        rag_chain,
        chatbot_obj.get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    cl.user_session.set("runnable", conversational_rag_chain_runnable)

    await cl.Message(
        content=f"Hi there, I am Alice, your AI powered chatbot for {company_name}. Ask me anything about {company_name}."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """
    Handles incoming chat messages.
    """
    session_id = cl.user_session.get("id")
    if not session_id:
        session_id = str(uuid.uuid4())
        cl.user_session.set("id", session_id)

    try:
        runnable = cast(Runnable, cl.user_session.get("runnable"))
        msg = cl.Message(content="")
        async for chunk in runnable.astream(
            {"input": message.content},
            config=RunnableConfig(
                callbacks=[cl.LangchainCallbackHandler()],
                configurable={"session_id": session_id},
            ),
        ):
            logger.info(f"Type of chunk from runnable : {type(chunk)}")
            await msg.stream_token(chunk.get("answer", ""))
        logger.info(f"LM Response : {msg.content}")
        await msg.send()

        # Use Deployed API for conversation
        # response = await chat_api.send_message(
        #     message=message.content, session_id=session_id
        # )
        # logger.info(f"API Response: {response}")
        # # Update the message with the API response
        # llm_response = response.get("response", "No response received.")
        # await cl.Message(content=llm_response).send()

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        msg = cl.Message(content=error_msg)
        await msg.update()
        await cl.ErrorMessage(content=error_msg).send()
