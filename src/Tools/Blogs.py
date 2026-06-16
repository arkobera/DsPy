import dspy
import asyncio
# from print_utils import print
from typing import Optional, List
from pydantic import BaseModel, Field
from avl_tools import duckduckgo_search
import os

OPENROUTER = os.getenv("OPENROUTER")

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
    base: Blog = dspy.InputField(description="The base structure of the blog post")
    draft: Optional[str] = dspy.InputField(description="Content of an existing blog")
    content: str = dspy.OutputField(description="The full content of the blog post")


class Judge(dspy.Signature):
    """
    You are a critical reviewer that evaluates the quality of a blog post based on its content and structure.
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
                 agent1: str = 'openrouter/deepseek/deepseek-r1',
                 agent2: str = 'openrouter/deepseek/deepseek-r1',
                 num_samples: int = 3,
                 temp1:float = 0.7,
                 temp2:float = 0.7,
                 judge_agent: str = 'openrouter/deepseek/deepseek-r1',):
        
        self.num_samples = num_samples
        self.agent1 = agent1
        self.agent2 = agent2
        self.judge_agent = judge_agent
        self.temp1 = temp1
        self.temp2 = temp2

        self.base_blog = dspy.ChainOfThought(Base)
        self.base_blog.set_lm(lm=dspy.LM(self.agent1, temperature=self.temp1, api_key = OPENROUTER))
        self.adv_blog = dspy.ReAct(advance, tools=[duckduckgo_search], max_iters=2)
        self.adv_blog.set_lm(lm=dspy.LM(self.agent2, temperature=self.temp2, api_key = OPENROUTER))
        self.judge = dspy.Refine(
            module = dspy.ChainOfThought(Judge),
            N=3, reward_fn=check_score_goodness, threshold=1
            )
        self.judge.set_lm(lm=dspy.LM(self.judge_agent, temperature=0.0, api_key = OPENROUTER))
        self.reflection = 2
        
    async def aforward(self, query):
        predictions = await asyncio.gather(*[self.base_blog.aforward(query=query) for _ in range(self.num_samples)])
        print("Gen")
        blog_ideas = [p.base for p in predictions]
        judge_score = self.judge(base=blog_ideas).rank 
        best_idea = judge_score.index(1)
        selected_idea = blog_ideas[best_idea]
        blog = None
        for _ in range(self.reflection):
            blog = self.adv_blog(base=selected_idea, draft=blog).content
        return blog
    
async def main():
    evaluator = ConditionalEvaluator()
    query = "The impact of artificial intelligence on modern education"
    blog_content = await evaluator.aforward(query=query)
    print(blog_content)

if __name__ == "__main__":
    asyncio.run(main())



