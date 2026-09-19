import arxiv
from app.core.logger import setup_logger

logger = setup_logger(__name__)

def fetch_papers(query: str, max_results: int) -> list[dict]:
    try:
        # Fetch papers from the arxiv api, sort them by relevance to the query
        # and return them as a list of dictionaries
        client = arxiv.Client()

        logger.info(f"Fetching {max_results} results for the query: {query}")
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        raw_results = client.results(search)

        results = []

        for raw_result in raw_results:
            results.append({
                "title": raw_result.title,
                "abstract": raw_result.summary,
                "authors": [r.name for r in raw_result.authors],
                "published": raw_result.published.strftime("%Y-%m-%d"),
                "url": raw_result.pdf_url,
                "categories": raw_result.categories
            })

        logger.info(f"Successfully fetched {len(results)} papers for query: '{query}'")

        return results

    except Exception as e:
        logger.error(f"Failed to fetch papers for query '{query}': {e}", exc_info=True)
        raise