from typing import List
from ddgs import DDGS

def duckduckgo_search(query: str, max_results: int = 5) -> List:
    """
    Inputs a query string, searches for news, and returns the top results.

    Args:
    query: String to search

    Returns:
    content: List of strings, each containing a news article about the topic
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    if not results:
        print("NO RESULTS")
        return []

    return [r['body'] for r in results]

if __name__ == "__main__":
    query = "What is an AVL tree?"
    print(duckduckgo_search(query))




