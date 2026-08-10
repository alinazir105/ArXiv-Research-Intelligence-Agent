from fastapi import FastAPI, HTTPException
from app.models.chat_request import ChatRequest
from app.agent import run_agent
from fastapi.concurrency import run_in_threadpool

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Arxiv Agent is running"}

@app.post("/chat")
async def call_agent(request: ChatRequest):
    try:
        result = await run_in_threadpool(run_agent, request.query)
        return {"answer" : result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))