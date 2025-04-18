"""
FastAPI server for model inference and task enqueuing.
"""
import os
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from celery import Celery
from dotenv import load_dotenv

from src_model import generate_response
from src_logger import log_event

# Load environment variables
load_dotenv()

# Configuration
PORT = int(os.getenv("API_PORT", "8000"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Celery instance for background tasks
celery_app = Celery("continuous_learning", broker=REDIS_URL)

# API models
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    prompt: str
    topic: str = "general"
    max_new_tokens: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    qid: str
    answer: str
    tokens: Optional[int] = None

# Create FastAPI app
app = FastAPI(
    title="Continuous Learning Lab",
    description="API for inference and continuous learning of LLMs",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "status": "online",
        "model": os.getenv("MODEL", "unknown"),
        "quantization": f"{os.getenv('QUANT_BITS', '4')}-bit"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Generate a response and queue it for teacher evaluation.
    """
    # Generate unique ID for tracking
    qid = uuid.uuid4().hex
    
    try:
        # Generate response from student model
        answer = generate_response(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature
        )
        
        # Enqueue for evaluation
        celery_app.send_task(
            "tasks.evaluate_response",
            kwargs={
                "prompt": request.prompt,
                "answer": answer,
                "qid": qid,
                "topic": request.topic
            }
        )
        
        # Log the event
        tokens = len(answer.split())
        log_event(
            "response_generated", 
            qid=qid, 
            tokens=tokens, 
            topic=request.topic
        )
        
        return ChatResponse(qid=qid, answer=answer, tokens=tokens)
    
    except Exception as e:
        log_event("generation_error", qid=qid, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src_server:app", host="0.0.0.0", port=PORT, reload=False)
