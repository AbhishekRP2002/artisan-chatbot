from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .rag_core import ChatbotRAG
from datetime import datetime
import uvicorn
import os
from chainlit.utils import mount_chainlit

chatbot = ChatbotRAG()

app = FastAPI(
    title="Artisan Support Chatbot",
    description="REST API for the Artisan Support Chatbot",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    message: str
    session_id: str = "default_session"


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        query = message.message
        session_id = message.session_id
        response = chatbot.get_response(query=query, session_id=session_id)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.get("/check-health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.now()}


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "code": f"ERROR_{exc.status_code}"},
    )


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAINLIT_APP_PATH = os.path.join(SCRIPT_DIR, "chainlit_app.py")

mount_chainlit(app=app, target=CHAINLIT_APP_PATH, path="/chainlit")

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
