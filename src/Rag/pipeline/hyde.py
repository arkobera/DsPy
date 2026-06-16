import dspy
from typing import Optional

from bm25_ret import BM25Retriever
from basic_rag import BasicEmbeddingsRAG
from rank_fus import reciprocal_rank_fusion

import os

from rich.console import Console
console = Console()


class HypotheticalDoc(dspy.Signature):
    """
    Given a query, generate hypothetical documents to search a database of one-liner blogs.
    """
    query: str = dspy.InputField(desc="User wants to fetch blogs related to this topic")
    retrieved_blogs: Optional[list[str]] = dspy.InputField(
        desc="blogs previously retrieved from the db. Use these to further tune your search."
    )

    hypothetical_bm25_query: str = dspy.OutputField(
        desc="sentence to query to retrieve more blogs about the query from the database"
    )
    hypothetical_semantic_query: str = dspy.OutputField(
        desc="sentence to search with cosine similarity"
    )


class MultiHopHydeSearch(dspy.Module):
    def __init__(self, texts, embs, n_hops=3, k=10):
        self.predict = dspy.ChainOfThought(HypotheticalDoc)
        self.predict.set_lm(lm=dspy.LM("groq/llama-3.1-8b-instant",api_key=os.getenv('GROQ_API')))
        self.embedding_retriever = BasicEmbeddingsRAG(texts, embs)
        self.bm25_retriever = BM25Retriever(texts)

        self.n_hops = n_hops
        self.k = k

    def forward(self, query):
        retrieved_blogs = []
        all_blogs = []
        for _ in range(self.n_hops):
            new_query = self.predict(query=query, retrieved_blogs=retrieved_blogs)
            print(new_query)
            embedding_lists = self.embedding_retriever.get_nearest(
                new_query.hypothetical_semantic_query
            )
            bm25_lists = self.bm25_retriever.get_nearest(
                new_query.hypothetical_bm25_query
            )
            lists = [embedding_lists, bm25_lists]
            retrieved_blogs = reciprocal_rank_fusion(lists, k=self.k)
            all_blogs.extend(retrieved_blogs)
        return dspy.Prediction(blogs=all_blogs)


if __name__ == "__main__":
    import numpy as np

    query = "Tourism"
    run_id = "1"
    k = 5
    n_hops = 3

    print(f"loading data for run_id: {run_id}...")
    with open(f"src/Rag/archive/blogs_{run_id}.txt", "r") as f:
        blogs = [line.strip() for line in f.readlines()]
    embeddings = np.load(f"src/Rag/archive/embeddings_{run_id}.npy")
    print("data loaded.")

    hyde = MultiHopHydeSearch(texts=blogs, embs=embeddings, n_hops=n_hops, k=k)
    retrieved_blogs = hyde(query=query).blogs
    console.print(retrieved_blogs)