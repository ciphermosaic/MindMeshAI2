
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm import generate_answer
from memory import (
    get_recent_messages,
    save_message,
    get_long_term_memory,
    remember,
)
from rag_retriever import retrieve_context

load_dotenv()

app = FastAPI(title="MindMeshAI Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://localhost:5137",
    ).split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    user_id: str = Field(min_length=1, max_length=128)


@app.get("/health")
def health():
    return {"status": "ok", "service": "MindMeshAI Chatbot"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    message = request.message.strip()
    user_id = request.user_id.strip()

    if not message:
        raise HTTPException(400, "Message cannot be empty.")

    if message.lower().startswith("remember:"):
        fact = message[len("remember:"):].strip()

        if not fact or len(fact) > 300:
            raise HTTPException(400, "Memory must be 1 to 300 characters.")

        remember(user_id, fact)
        return {"answer": "I'll remember that for future conversations."}

    history = get_recent_messages(user_id)
    memories = get_long_term_memory(user_id)

    try:
        context = await retrieve_context(message, user_id)

        answer = generate_answer(
            message,
            context=context,
            history=history,
            memories=memories,
        )

        save_message(user_id, "user", message)
        save_message(user_id, "assistant", answer)

        return {"answer": answer}

    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=502,
            detail=f"Chatbot failed: {type(e).__name__}: {e}",
        )
