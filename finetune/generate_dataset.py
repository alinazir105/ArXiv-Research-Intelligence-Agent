import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
from qdrant_client import QdrantClient
from openai import OpenAI
from app.core.config import settings
import json

qdrant = QdrantClient(url=settings.QDRANT_URL)
openai = OpenAI(api_key=settings.OPENAI_API_KEY)

output_path = os.path.join(os.path.dirname(__file__), "dataset.jsonl")

def generate_finetune_dataset():
    all_points = fetch_all_chunks(qdrant)

    for i, point in enumerate(all_points):
        try:
            chunk_text = point.payload["text"]

            # Generate questions using OpenAI API
            SYSTEM_PROMPT = f"""
                Given this research paper excerpt, generate 2 focused questions that this text directly answers.
                Return ONLY the questions, one per line, no numbering, no preamble.

                Text: {chunk_text}
            """
            response = openai.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                   {
                        "role": "user",
                        "content": SYSTEM_PROMPT
                    }
                ]
            )
        
            questions = response.choices[0].message.content

            # Parse questions into a list
            questions_list = [q.strip() for q in questions.splitlines() if q.strip()]

            # Saving question answer pair to dataset.jsonl
            with open(output_path, "a") as f:
                for question in questions_list:
                    record = {
                        "question": question,
                        "answer": chunk_text
                    }
                    f.write(json.dumps(record) + "\n")

            # Sleep for a short duration to avoid hitting rate limits
            time.sleep(0.5)

            # Print progress every 50 chunks
            if i % 50 == 0:
                print(f"Processed {i}/{len(all_points)} chunks.")

        except Exception as e:
            print(f"Chunk {i} failed: {e}")
            continue


def fetch_all_chunks(client: QdrantClient) -> list:
    """Fetch all points from Qdrant synchronously."""
    all_points = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name="arxiv_papers",
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(records)
        if offset is None:
            break
    return all_points


if __name__ == "__main__":
    generate_finetune_dataset()