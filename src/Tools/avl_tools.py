# from duckduckgo_search import DDGS
from ddgs import DDGS

def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """
    Search DuckDuckGo and return summarized results.
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    if not results:
        return "No results found."

    return "\n\n".join(
        f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
        for r in results
    )

if __name__ == "__main__":
    query = "What is an AVL tree?"
    print(duckduckgo_search(query))

