from app.core.config import settings
from qdrant_client import AsyncQdrantClient
from openai import AsyncOpenAI
from qdrant_client.models import Record
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder
import asyncio
from app.core.logger import setup_logger

logger = setup_logger(__name__)

class HybridRetriever:
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL)
        self.cross_encoder = None
        self.chunks = None
        self.chunk_texts = None
        self.bm25 = None


    @classmethod
    async def create(cls):
        """Asynchronous factory method to create an instance of HybridRetriever."""
        self = cls()

        # cross-encoder is loaded once at startup — loading it per query would add
        # several seconds of latency every time. It runs locally, no API cost.
        logger.info("Loading cross-encoder model...")
        self.cross_encoder = await asyncio.to_thread(CrossEncoder, "cross-encoder/ms-marco-MiniLM-L-6-v2")

        logger.info(f"Fetching all chunks from Qdrant...")
        all_points = await fetch_records_from_qdrant(self.qdrant_client)

        # store full payloads for metadata lookup during BM25 search —
        # BM25 works on indices, so we need the original chunks to map back to titles/urls
        self.chunks = [point.payload for point in all_points]
        self.chunk_texts = [point.payload["text"] for point in all_points]

        logger.info(f"Loaded {len(all_points)} chunks, building BM25 index...")
        # BM25 index is built once at startup from all chunks in the corpus.
        # rebuilding it on every query would require fetching all points from Qdrant
        # and reprocessing them — unacceptable latency at query time.
        # lowercase + split is enough tokenization for BM25 — it's keyword matching,
        # not semantic, so we just need consistent token boundaries.
        tokenized = [text.lower().split() for text in self.chunk_texts]
        self.bm25 = await asyncio.to_thread(BM25Okapi, tokenized)
        logger.info("HybridRetriever ready.")

        return self

    async def _bm25_search(self, query: str, k: int) -> list[dict]:
        """Score all chunks with BM25, return top-k with index and score."""
        # tokenize the same way as the corpus — consistency is what matters,
        # not sophistication. mismatched tokenization would tank recall.
        tokenized_query = query.lower().split()
        
        # get_scores returns one float per chunk in the corpus — not ranked, just scored
        scores = await asyncio.to_thread(self.bm25.get_scores, tokenized_query)

        # argsort gives ascending order, [::-1] reverses to descending,
        # [:k] takes the top k indices
        top_indices = np.argsort(scores)[::-1][:k]

        top_chunks_with_metadata = [
             {
                  "text": self.chunks[i]["text"],
                  "title": self.chunks[i]["title"],
                  "url": self.chunks[i]["url"],
                  # scores[i] is the raw BM25 float for this chunk —
                  # not comparable to cosine scores, only meaningful relative to other BM25 scores
                  "score": scores[i]
             }
             for i in top_indices
        ]

        return top_chunks_with_metadata

    async def _rrf_fusion(self, dense_results, bm25_results, k: int) -> list[dict]:
        """Combine dense and BM25 results using Reciprocal Rank Fusion."""

        # keyed by chunk text so the same chunk from both lists gets its scores accumulated,
        # not stored as two separate entries
        rrf_scores = {}

        for i, result in enumerate(dense_results):
            # k=60 is the standard RRF constant from the original paper —
            # it dampens the impact of very high ranks so position 1 isn't
            # disproportionately rewarded over position 2
            contribution = 1 / (60 + i)
            if result["text"] not in rrf_scores:
                rrf_scores[result["text"]] = {
                    "score": contribution,
                    "title": result["title"],
                    "url": result["url"]
                }
            else:
                # same chunk appeared in both lists — accumulate both contributions.
                # this is what rewards chunks that rank well in multiple sources.
                rrf_scores[result["text"]]["score"] += contribution

        for i, result in enumerate(bm25_results):
            contribution = 1 / (60 + i)
            if result["text"] not in rrf_scores:
                rrf_scores[result["text"]] = {
                    "score": contribution,
                    "title": result["title"],
                    "url": result["url"]
                }
            else:
                rrf_scores[result["text"]]["score"] += contribution

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)

        return [
            {
                "text": text,
                "title": data["title"],
                "url": data["url"],
                "score": data["score"]
            }
            for text, data in sorted_results[:k]
        ]
    
    async def _rerank(self, query: str, results: list[dict], k: int) -> list[dict]:
        """Score query-chunk pairs with cross-encoder, return top-k."""

        # cross-encoder sees query and chunk together in one pass —
        # unlike bi-encoder which embeds them separately and compares vectors.
        # this catches relevance signals that embedding comparison misses,
        # but it's too slow to run on the full corpus — only on top candidates.
        pairs = [[query, result["text"]] for result in results]
        scores = await asyncio.to_thread(self.cross_encoder.predict, pairs)

        # attach scores back to results and sort
        for i, result in enumerate(results):
            result["score"] = scores[i]

        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        return sorted_results[:k]

    async def _generate_hypothetical_document(self, query: str) -> str:
        """Generate a fake academic abstract for the query using the LLM."""
        try:
            # HyDE: instead of embedding the user's casual query directly,
            # we generate a fake abstract written in academic language.
            # academic text clusters with academic text in vector space —
            # so the fake abstract lands closer to real ArXiv papers than
            # the user's casual phrasing would.
            # factual accuracy doesn't matter here — only stylistic accuracy.
            SYSTEM_PROMPT = f"""
                You are a HyDE agent. Your job is to convert the given query into a sample research paper abstract,
                written in formal academic language with technical terminology, as it would appear on ArXiv.
                Do not answer the question — only write the abstract.

                Query: {query}
            """
            response = await self.openai_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": SYSTEM_PROMPT
                    }
                ]
            )

            hypothetical_doc = response.choices[0].message.content
            return hypothetical_doc

        except Exception as e:
            logger.error(f"HyDE failed: {e}", exc_info=True)
            raise

    async def _decompose_query(self, query: str) -> list[str]:
        """Breaks the query down into focused sub-questions."""
        try:
            # a single query vector can't represent multiple intents simultaneously —
            # it gets pulled in multiple directions and represents none well.
            # decomposing into sub-questions lets each one pull its own focused chunks.
            SYSTEM_PROMPT = f"""
                You are a Query Decomposition agent. Your job is simply to break down a query into 2-3 focused
                sub-questions. Return the questions as a numbered list. If the query is simple already then just
                return that query as a single item list.

                Query: {query}
            """
            response = await self.openai_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": SYSTEM_PROMPT
                    }
                ]
            )

            decomposed_query = response.choices[0].message.content

            # strip numbering and whitespace from each line to get clean sub-questions
            lines = decomposed_query.strip().split("\n")
            sub_questions = [
                line.lstrip("0123456789.-) ").strip()
                for line in lines
                if line.strip()
            ]
            return sub_questions

        except Exception as e:
            logger.error(f"Query Decomposition failed: {e}", exc_info=True)
            # fallback to original query so retrieval still works if decomposition fails
            return [query]

    async def retrieve(self, query: str, k: int = 5):
        try:
            # 20 candidates per stage gives the cross-encoder enough to work with
            # without becoming slow. too few and the best chunk might not make it through.
            CANDIDATES = 20
            
            sub_questions = await self._decompose_query(query=query)
            logger.debug(f"Decomposed into {len(sub_questions)} sub-questions")
            
            all_results = []

            for sub_question in sub_questions:
                # embed the hypothetical doc, not the raw sub-question —
                # academic-style text lands closer to ArXiv abstracts in vector space
                hypothetical_doc = await self._generate_hypothetical_document(query=sub_question)

                response = await self.openai_client.embeddings.create(
                    input=[hypothetical_doc],
                    model=settings.EMBEDDING_MODEL
                )

                sub_question_vector = response.data[0].embedding

                response = await self.qdrant_client.query_points(
                    collection_name=settings.COLLECTION_NAME,
                    query=sub_question_vector,
                    limit=CANDIDATES
                )
                raw_results = response.points

                dense_search_results = [
                    {
                        "text": raw_result.payload["text"],
                        "title": raw_result.payload["title"],
                        "url": raw_result.payload["url"],
                        "score": raw_result.score
                    }
                    for raw_result in raw_results
                ]

                bm25_search_results = await self._bm25_search(query=sub_question, k=CANDIDATES)

                rrf_results = await self._rrf_fusion(
                    dense_results=dense_search_results,
                    bm25_results=bm25_search_results,
                    k=20
                )
                
                all_results.extend(rrf_results)

            # deduplicate by chunk text — the same chunk may surface across
            # multiple sub-question retrievals. we keep first occurrence
            # (highest RRF score) and discard duplicates.
            seen = set()
            unique_results = []
            for result in all_results:
                if result["text"] not in seen:
                    seen.add(result["text"])
                    unique_results.append(result)

            # rerank with the ORIGINAL query, not sub-questions —
            # the cross-encoder judges relevance to what the user actually asked,
            # not to the retrieval decomposition we used internally
            reranked_results = await self._rerank(query=query, results=unique_results, k=k)
            logger.info(f"Retrieved {len(reranked_results)} results for query: '{query[:50]}'")
            return reranked_results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            raise


async def fetch_records_from_qdrant(qdrant_client: AsyncQdrantClient) -> list[Record]:
    """Fetch all points from Qdrant using pagination."""
    all_points = []
    offset = None

    while True:
        # scroll paginates through the collection in pages of 100 —
        # fetching all points in one call would risk memory issues at scale
        response = await qdrant_client.scroll(
            collection_name=settings.COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            # vectors not needed here — we only need text and metadata
            # for BM25 index and chunk lookup. skipping vectors saves memory.
            with_vectors=False,
        )
        records, offset = response
        
        all_points.extend(records)

        logger.debug(f"Fetched page, total so far: {len(all_points)}")
        
        # offset is None when Qdrant has no more pages to return
        if offset is None:
            break

    logger.info(f"Fetched {len(all_points)} total points from Qdrant")
    return all_points