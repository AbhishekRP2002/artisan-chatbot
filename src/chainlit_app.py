import chainlit as cl
import httpx
import json  # noqa
from typing import Dict
from dotenv import load_dotenv
import uuid
import logging

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


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Can Ava access my CRM?",
            message="Can Ava access my CRM?",
        ),
        cl.Starter(
            label="Ava, the Top-Rated AI SDR on the market",
            message="How can Ava help in automating my SDR workflows in my sales pipelines or my outbound demand generation process?",
        ),
        cl.Starter(
            label="Create a Campaign",
            message="How can Artisan help in creating a campaign to engage potential leads effectively?",
        ),
        cl.Starter(
            label="Generate Sample Email",
            message="Explain the 'Generate Sample Email feature and how to use it via the Artisan Platform.",
        ),
    ]


# @cl.on_chat_start
# async def start():
#     """
#     Initializes the chat session.
#     """
#     await cl.Message(
#         content="👋 Hello! I'm your Artisan AI Support Assistant. How can I help you today?",
#     ).send()


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
        response = await chat_api.send_message(
            message=message.content, session_id=session_id
        )
        logger.info(f"API Response: {response}")
        # Update the message with the API response
        llm_response = response.get("response", "No response received.")
        await cl.Message(content=llm_response).send()

        # If source references are available, append them as additional elements
        # metadata = response.get("metadata", {})
        # if "sources" in metadata:
        #     sources = metadata["sources"]
        #     elements = [
        #         cl.Text(name=f"Source {i+1}", content=source)
        #         for i, source in enumerate(sources)
        #     ]
        #     await msg.update(elements=elements)

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        msg = cl.Message(content=error_msg)
        await msg.update()
        await cl.ErrorMessage(content=error_msg).send()
