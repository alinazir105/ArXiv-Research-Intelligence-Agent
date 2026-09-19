from fastapi import FastAPI, HTTPException
from app.core.cache import get_cached, set_cached
from app.models.chat_request import ChatRequest
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from app.core.config import settings
import json
from contextlib import asynccontextmanager
from app.agent.agent import initialize, run_agent
from app.core.logger import setup_logger
from qdrant_client import QdrantClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

logger = setup_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ArXiv Research Intelligence Agent...")
    await initialize()
    logger.info("Application startup complete.")
    yield
    logger.info("Application shutting down.")

app = FastAPI(
    lifespan=lifespan, 
    title="Arxiv Agent", 
    description="An AI research assistant for ArXiv papers.", 
    version="1.0.0"
    )

# create limiter - identifies users by their IP address
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/")
def read_root():
    return {"message": "Arxiv Agent is running"}

@app.post("/chat")
@limiter.limit("10/minute")  # limit to 10 requests per minute per IP
async def call_agent(request: Request, body: ChatRequest):
    try:
        # check cache first
        cached = await get_cached(body.query)
        if cached:
            return {"answer": cached["answer"], "cached": True}
        result = await run_agent(body.query)
        await set_cached(body.query, result)
        return {"answer" : result["answer"], "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
@limiter.limit("10/minute")  # limit to 10 requests per minute per IP
async def stream_agent(request: Request, body: ChatRequest):
    async def generate():
        try:
            # step 1: get context from agent tools (blocking, run in thread)
            result = await run_agent(body.query)
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

                                Question: {body.query}"""
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

@app.get("/health")
async def health_check():
    health = {
        'status': 'healthy',
        'qdrant' : 'unknown',
        'retriever': 'unknown'
    }

    # Check Qdrant
    try:
        client = QdrantClient(url=settings.QDRANT_URL)
        client.get_collections()
        health['qdrant'] = 'ok'
        logger.info("Qdrant health check passed.")
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}", exc_info=True)
        health['qdrant'] = f'error: {str(e)}'
        health['status'] = 'degraded'

    # Check Retriever
    try:
        from app.agent.agent import retriever
        if retriever is not None:
            health['retriever'] = 'ok'
            logger.info("Retriever health check passed.")
        else:
            health['retriever'] = 'not initialized'
            health['status'] = 'degraded'
    except Exception as e:
        logger.error(f"Retriever health check failed: {e}", exc_info=True)
        health['retriever'] = f'error: {str(e)}'
        health['status'] = 'degraded'

    return health