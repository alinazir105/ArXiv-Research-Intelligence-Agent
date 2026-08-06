from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
import uuid
from app.core.config import settings
import hashlib

def get_clients():
    # Fetch the OpenAI and Qdrant clients
    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    qdrant_client = QdrantClient(url=settings.QDRANT_URL)

    return openai_client, qdrant_client

def ensure_collection(qdrant_client: QdrantClient):
    # Check that the collection already exists or not. If not, then create it
    existing_collections = [c.name for c in qdrant_client.get_collections().collections]
    if settings.COLLECTION_NAME not in existing_collections:
        qdrant_client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=settings.VECTOR_DIMENSION, 
                distance=Distance.COSINE
                )
        )

def generate_chunk_id(text: str) -> str:
    # Create a hashed id, to avoid duplicate chunks in Qdrant. 
    # Duplicate chunks occur if you run the pipeline again for the same corpus. Without a hashed id, each chunk will be assigned a new id.
    # A hash for the same chunk will always be the same, hence the same id will be generated, so duplication will be avoided
    hash_hex = hashlib.md5(text.encode()).hexdigest()
    return str(uuid.UUID(hash_hex))

def index_papers(chunks: list[dict]):
    # Call the OpenAI api to convert chunks into vectors. Store the vectors and metadata along with the hashed id in qdrant
    try:
        # Get the clients for Open AI and Qdrant
        openai_client, qdrant_client = get_clients()

        # Ensure that the collection exisits if it doesnt, create a new one
        ensure_collection(qdrant_client)

        # Extract the chunk texts into a list
        text_to_embed = [chunk["text"] for chunk in chunks]

        # Call the Open AI api to convert text to embeddings
        response = openai_client.embeddings.create(
            input=text_to_embed,
            model=settings.EMBEDDING_MODEL
        )

        # Extract the embeddings for each chunk into a list
        embeddings = [data.embedding for data in response.data]

        # Create Qdrant Points for each chunk
        points = [
            PointStruct(
                    id=generate_chunk_id(chunk['text']),
                    vector=embeddings[i],
                    payload={
                        "text": chunk['text'],
                        "title": chunk['title'],
                        "authors": chunk['authors'],
                        "published": chunk['published'],
                        "url": chunk['url'],
                        "categories": chunk['categories']
                    }
                )
            for i, chunk in enumerate(chunks)
        ]

        # Inert or update the points
        qdrant_client.upsert(
            collection_name=settings.COLLECTION_NAME,
            wait=True,
            points=points
        )

    except Exception as e:
        print(f"Indexing failed: {e}")
        raise

