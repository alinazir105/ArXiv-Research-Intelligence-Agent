from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator
from langchain_core.tools import tool
from app.retrieval.retriever import HybridRetriever
from ddgs import DDGS
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from app.core.config import settings
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
import arxiv
import os
import asyncio
from app.core.logger import setup_logger

logger = setup_logger(__name__)

os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

class AgentState(TypedDict):
    messages : Annotated[list[BaseMessage], operator.add]


retriever = None
web_search = DDGS()

async def initialize():
    """Initialize async resources — called once at application startup."""
    global retriever # access the global retriever variable
    logger.info("Initializing HybridRetriever...")
    retriever = await HybridRetriever.create()
    logger.info("Agent initialized successfully.")

# The docstring are necessary for tool nodes as the LLm reads them to decide when to use each tool
@tool
async def search_corpus(query: str) -> str:
    """Search the ArXiv research paper corpus for AI/ML research concepts."""
    try:
        logger.info(f"Searching corpus for: '{query[:50]}'")
        results = await retriever.retrieve(query=query)
        return "\n\n".join([
            f"Title: {r['title']}\n{r['text']}"
            for r in results
        ])
    except Exception as e:
        logger.error(f"Corpus search failed: {e}", exc_info=True)
        return f"Corpus search failed: {e}"

@tool
async def search_web(query: str) -> str:
    """Search the web for current information not in the research corpus."""
    try:
        logger.info(f"Searching web for: '{query[:50]}'")
        results = await asyncio.to_thread(web_search.text, query, max_results=5)
        return "\n\n".join([r["body"] for r in results])
    except Exception as e:
        logger.error(f"Web search failed: {e}", exc_info=True)
        return f"Web search failed: {e}"

@tool
async def fetch_paper(url: str) -> str:
    """Fetch full details of a specific ArXiv paper when you have its URL.
    Use this when you need more detail about a paper already identified
    through search. Do not use for general topic searches — use 
    search_corpus instead."""
    try:
        def _fetch():
            paper_id = url.split("/")[-1]
            client = arxiv.Client()
            search = arxiv.Search(id_list=[paper_id])
            return next(client.results(search))

        # Run the blocking, synchronous _fetch function in a separate thread 
        # to avoid freezing the main async event loop
        logger.info(f"Fetching the full paper from: {url}")
        paper = await asyncio.to_thread(_fetch)
        return f"Title: {paper.title}\n\nAuthors: {', '.join([a.name for a in paper.authors])}\n\nAbstract: {paper.summary}\n\nURL: {paper.entry_id}"
    except Exception as e:
        logger.error(f"Paper fetch failed: {e}", exc_info=True)
        return f"Paper fetch failed: {e}"


@tool
async def summarize_papers(query: str) -> str:
    """Retrieve and summarize papers on a topic from the ArXiv corpus.
    Use this when the user asks to summarize, compare, or get an overview 
    of papers on a specific topic. Not for fetching a specific paper by URL 
    — use fetch_paper for that."""
    try:
        logger.info(f"Fetching paper summaries for the query: '{query[:50]}'")
        results = await retriever.retrieve(query=query, k=5)
        summaries = []
        for r in results:
            summaries.append(f"Title: {r['title']}\nSummary: {r['text']}")
        return "\n\n".join(summaries)
    except Exception as e:
        logger.error(f"Paper summarization failed: {e}", exc_info=True)
        return f"Paper summarization failed: {e}"


# bind tools to the LLM so it knows what functions it can call
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key= settings.OPENAI_API_KEY
)
tools = [search_corpus, search_web, fetch_paper, summarize_papers]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_MESSAGE = SystemMessage(content="""You are a research assistant with access to tools.
Always use your tools to find information before answering.
Base your answers ONLY on what your tools return — never answer from your own knowledge.
If your tools don't return relevant information, say so clearly.""")

# agent node — LLM thinks and decides: call a tool or answer
async def agent_node(state: AgentState) -> dict:
    """LLM reads message history and decides next action."""
    response = await llm_with_tools.ainvoke([SYSTEM_MESSAGE] + state['messages'])
    return {'messages' : [response]}

# tools node — executes whichever tool the agent called
tool_node = ToolNode(tools)

def should_continue(state: AgentState) -> str:
    """Route to tools if agent made a tool call, otherwise end."""
    last_message = state['messages'][-1]

    if last_message.tool_calls:
        return "tools"

    return "end"

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )

    graph.add_edge("tools", "agent")

    return graph.compile()

agent_graph = build_graph()

async def run_agent(query: str) -> dict:
    """Run the agent with a user query and return the final answer."""
    try:
        logger.info(f"Running agent for query: '{query[:50]}'")
        result = await agent_graph.ainvoke({
            "messages": [HumanMessage(content=query)]
        })


        # find tool use message
        tool_message = next((m for m in result['messages'] if hasattr(m, 'tool_calls') and m.tool_calls), None)

        if tool_message is None:
            logger.info("Agent completed without using any tools.")
            return {
                "answer": result["messages"][-1].content,
                "tool_used": "none",
                "context": ""
            }

        tool_used = tool_message.tool_calls[0]['name']
        logger.info(f"Agent completed. Tool used: {tool_used}")

        # find tool result message
        tool_result = next(m for m in result['messages'] if isinstance(m, ToolMessage))
        context = tool_result.content

        # last message is the agent's final text response
        answer = result["messages"][-1].content

        return {
            "answer": answer,
            "tool_used": tool_used,
            "context": context
        }

    except Exception as e:
        logger.error(f"Agent failed: {e}", exc_info=True)
        raise RuntimeError(f"Agent failed: {e}")

if __name__ == "__main__":
    asyncio.run(initialize())
    answer = asyncio.run(run_agent("What are the differences between RAG and fine-tuning?"))
    print(answer)