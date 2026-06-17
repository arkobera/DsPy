import dspy
import asyncio
# from print_utils import print
from typing import Optional, List
from pydantic import BaseModel, Field
from hyde import MultiHopHydeSearch
from tools import duckduckgo_search
import os

import numpy as np

GROQ_API = os.getenv("GROQ_API")
OPENROOUTER_API = os.getenv('OPENROUTER')
GOOGLE_API= os.getenv('GOOGLE_API_KEY')

GROQ_MODEL = 'groq/llama-3.1-8b-instant'
GOOGLE_MODEL = 'gemini/'


MODEL = GROQ_MODEL
API = GROQ_API

dspy.configure(track_usage=True)
dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
)

class Blog(BaseModel):
    topic: str
    intro: str
    body: str
    conclusion: str
    

class Base(dspy.Signature):
    """
    You are a helpful assistant that writes blog posts based on the given topic, length, and style.
    """
    query: str = dspy.InputField(description="The topic of the blog post")
    base: Blog = dspy.OutputField(description="The base structure of the blog post")


class advance(dspy.Signature):
    """
    You are a creative writer that takes the base structure of a blog post and generates the full content.
    """
    base: str = dspy.InputField(description="The base structure of the blog post")
    draft: Optional[str] = dspy.InputField(description="Content of an existing blog")
    content: str = dspy.OutputField(description="The full content of the blog post")


class Judge(dspy.Signature):
    """
    You are a critical reviewer that evaluates the quality of a blog post based on its content and structure.
    Rank 1 is the best Blog.
    """
    base: List[Blog] = dspy.InputField(description="The base structure of the blog post")
    rank: List[int] = dspy.OutputField(description="The quality score of the blog post 1, 2, 3, .. N")


def check_score_goodness(args, pred):
    num_samples = len(args['base'])
    same_length = len(pred.rank) == num_samples
    all_rank = all([(i+1) in pred.rank for i in range(num_samples)])
    return 1 if same_length and all_rank else 0


class ConditionalEvaluator(dspy.Module):
    def __init__(self,
                 retriever,
                 agent1: str = MODEL,
                 agent2: str = MODEL,
                 num_samples: int = 1,
                 temp1:float = 1,
                 temp2:float = 1,
                 judge_agent: str = MODEL,):
        
        self.num_samples = num_samples
        self.agent1 = agent1
        self.agent2 = agent2
        self.judge_agent = judge_agent
        self.temp1 = temp1
        self.temp2 = temp2

        self.base_blog = dspy.ReAct(Base, tools=[duckduckgo_search],max_iters=3)
        self.base_blog.set_lm(lm=dspy.LM(self.agent1, temperature=self.temp1, api_key = GROQ_API))
        self.adv_blog = dspy.ChainOfThought(advance)
        self.adv_blog.set_lm(lm=dspy.LM(self.agent2, temperature=self.temp2, api_key = GROQ_API))
        self.judge = dspy.Refine(
            module = dspy.ChainOfThought(Judge),
            N=3, reward_fn=check_score_goodness, threshold=1
            )
        self.judge.set_lm(lm=dspy.LM(self.judge_agent, temperature=0.0, api_key = GROQ_API))
        self.reflection = 2
        self.retriever = retriever
        
    async def aforward(self, query):
        predictions = await asyncio.gather(*[self.base_blog.aforward(query=query) for _ in range(self.num_samples)])
        print(predictions, type(predictions))
        # blog_ideas = [p.base for p in predictions]
        blog_ideas = predictions
        # judge_score = self.judge(base=blog_ideas).rank 
        # best_idea = judge_score.index(1)
        # selected_idea = blog_ideas[best_idea]
        selected_idea = blog_ideas
        search_query = f"""
                    query = {query},
                    ideas = {selected_idea}
                        """
        rel_content = self.retriever(query=search_query).blogs
        # print(blog_ideas)
        # blog_ideas = [p.base for p in predictions]
        blog = None
        for _ in range(self.reflection):
            blog = self.adv_blog(base=rel_content, draft=blog).content
        return blog
    
async def main(query):

    ### Load Data
    run_id = "1"
    with open(f"src/Rag/archive/blogs_{run_id}.txt", "r") as f:
        blogs = [line.strip() for line in f.readlines()]
    embeddings = np.load(f"src/Rag/archive/embeddings_{run_id}.npy")

    ### Initialize the retriever
    retriever = MultiHopHydeSearch(blogs, embeddings, n_hops=2, k=5)
    
    evaluator = ConditionalEvaluator(retriever=retriever)
    blog_content = await evaluator.aforward(query=query)
    print(blog_content)

if __name__ == "__main__":
    query = "The impact of artificial intelligence on modern education"
    asyncio.run(main(query))



