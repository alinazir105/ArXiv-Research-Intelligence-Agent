import arxiv

def fetch_papers(query: str, max_results: int) -> list[dict]:
    # Fetch papers from the arxiv api, sort them by relevance to the query
    # and return them as a list of dictionaries
    client = arxiv.Client()
    
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

    return results
