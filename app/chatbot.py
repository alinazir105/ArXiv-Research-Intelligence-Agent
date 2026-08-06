from indexer import get_clients
from core.config import settings
from retriever import HybridRetriever

retriever = HybridRetriever()

def chat(query: str) -> dict:
    # Retrieve the k most relevant chunks to the query. Combine the query as question and chunks as context in the prompt
    # that will be sent to the OpenAI api. Return a dictionary with answer, and the sources used for the answer.
    try:
        openai_client, _ = get_clients()

        results = retriever.retrieve(query=query)

        SYSTEM_PROMPT = f"""
        You are a research assistant. Answer the question using ONLY the context below.
        If the answer is not in the context, say "I couldn't find relevant information."
        Synthesize information across multiple context passages where relevant.

        Context:
        {"\n\n".join([result["text"] for result in results])}

        Question: {query}
        """

        response = openai_client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": SYSTEM_PROMPT
                }
            ]
        )

        return{
            "answer": response.choices[0].message.content,
            "sources": list(set([r["title"] for r in results]))
        }

    except Exception as e:
        print(f"Chatbot failed: {e}")
        raise

if __name__ == "__main__":
    result = chat("What are the differences between RAG and fine-tuning, and when should I use each?")
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"  - {source}")