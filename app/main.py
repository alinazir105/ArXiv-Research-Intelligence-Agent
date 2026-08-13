from fastapi import FastAPI, HTTPException
from app.models.chat_request import ChatRequest
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from app.core.config import settings
import json
from contextlib import asynccontextmanager
from app.agent import initialize, run_agent

@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize()
    yield

app = FastAPI(
    lifespan=lifespan, 
    title="Arxiv Agent", 
    description="An AI research assistant for ArXiv papers.", 
    version="1.0.0"
    )

@app.get("/")
def read_root():
    return {"message": "Arxiv Agent is running"}

@app.post("/chat")
async def call_agent(request: ChatRequest):
    try:
        result = await run_agent(request.query)
        return {"answer" : result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def stream_agent(request: ChatRequest):
    async def generate():
        try:
            # step 1: get context from agent tools (blocking, run in thread)
            result = await run_agent(request.query)
            context = result["context"]

            # step 2: stream generation from OpenAI directly
            async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

            stream = await async_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": f"""You are a research assistant. Answer using ONLY the context below.
                                If the answer is not in the context, say so clearly.

                                Context:
                                {context}

                                Question: {request.query}"""
                }],
                stream=True
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'token': token})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
