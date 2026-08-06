from ingest import fetch_papers
from chunker import chunk_papers
from indexer import index_papers
import time

def run_pipeline(query: str, max_results: int):
    papers = fetch_papers(query=query, max_results=max_results)
    chunks = chunk_papers(papers=papers)
    index_papers(chunks=chunks)
    print(f"Indexed {len(chunks)} chunks from {len(papers)} papers")

if __name__ == "__main__":
    SEED_QUERIES = [
        "retrieval augmented generation",
        "large language model fine tuning",
        "transformer attention mechanism",
        "vector database semantic search",
        "prompt engineering LLM",
        "AI agent planning tool use",
        "diffusion models image generation",
        "reinforcement learning from human feedback",
        "knowledge graph reasoning",
        "multimodal learning vision language"
    ]

    for query in SEED_QUERIES:
        run_pipeline(query=query, max_results=50)
        time.sleep(3)  # wait 3 seconds between queries
