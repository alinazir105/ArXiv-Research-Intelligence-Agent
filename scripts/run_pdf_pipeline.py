from app.ingestion.ingest import fetch_papers
from app.ingestion.indexer import index_papers
import time
from app.core.config import settings
from openai import OpenAI
from app.ingestion.pdf_processor import download_pdf, extract_text, chunk_and_contextualize

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

def main():
    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    total_papers = 0
    total_chunks = 0

    for query in SEED_QUERIES:
        print(f"\nQuery: {query}")
        papers = fetch_papers(query, max_results=5)

        for paper in papers:
            try:
                print(f"  Processing: {paper['title'][:60]}...")

                # download and extract text from full PDF
                pdf_bytes = download_pdf(url=paper['url'])
                text = extract_text(pdf_bytes=pdf_bytes)

                if not text.strip():
                    print(f"  Skipping — no text extracted")
                    continue

                # chunk and add contextual prefixes
                chunks = chunk_and_contextualize(
                    text=text,
                    title=paper['title'],
                    openai_client=openai_client
                )

                # fill in paper metadata on each chunk
                for chunk in chunks:
                    chunk['url'] = paper['url']
                    chunk['authors'] = paper['authors']
                    chunk['published'] = paper['published']

                # index into Qdrant
                index_papers(chunks)

                total_papers += 1
                total_chunks += len(chunks)
                print(f"  Indexed {len(chunks)} chunks")

                # avoid hammering ArXiv
                time.sleep(1)

            except Exception as e:
                print(f"  Failed: {e}")
                continue

    print(f"\nDone. {total_papers} papers, {total_chunks} chunks indexed.")


if __name__ == "__main__":
    main()