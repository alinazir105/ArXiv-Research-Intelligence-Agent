from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ingestion.ingest import fetch_papers

def chunk_papers(papers: list[dict]) -> list[dict]:
    # Breakdown the abstracts from the papers into chunks, and store them into a list, 
    # with metadata for each chunk, and return the list
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100
    )

    chunks = []

    for paper in papers:
        documents = text_splitter.create_documents(
            texts=[paper['abstract']],
            metadatas=[
                {
                    "title": paper['title'],
                    "authors": paper['authors'],
                    "published": paper['published'],
                    "url": paper['url'],
                    "categories": paper['categories']
                }
            ]
        )

        chunks.extend([
            {
                "text": doc.page_content,
                "title": doc.metadata['title'],
                "authors": doc.metadata['authors'],
                "published": doc.metadata['published'],
                "url": doc.metadata['url'],
                "categories": doc.metadata['categories']
            }
            for doc in documents
        ])

    return chunks


if __name__ == "__main__":
    papers = fetch_papers("RAG retrieval augmented generation", 2)
    chunks = chunk_papers(papers)
    print(f"Papers: 2, Chunks: {len(chunks)}")
    print(chunks[0])