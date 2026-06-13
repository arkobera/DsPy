import time
import dspy
import asyncio

from typing import List, Optional
from pydantic import BaseModel, Field

import random
import pandas as pd

# import mlflow
# mlflow.autolog()
# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment("dspy-eval")

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
                 agent1: str,
                 agent2: str,
                 num_samples: int = 3,
                 temp1:float = 0.7,
                 temp2:float = 0.7,
                 judge_agent: str = 'phi3:mini',
                 api_base: str = "http://localhost:11434"):
        
        self.num_samples = num_samples
        self.agent1 = agent1
        self.agent2 = agent2
        self.judge_agent = judge_agent
        self.api_base = api_base
        self.temp1 = temp1
        self.temp2 = temp2

        self.base_blog = dspy.ChainOfThought(Base)
        self.base_blog.set_lm(lm=dspy.LM(self.agent1, temperature=self.temp1, api_base=self.api_base))
        self.adv_blog = dspy.ChainOfThought(advance)
        self.adv_blog.set_lm(lm=dspy.LM(self.agent2, temperature=self.temp2, api_base=self.api_base))
        self.judge = dspy.Refine(
            module = dspy.ChainOfThought(Judge),
            N=3, reward_fn=check_score_goodness, threshold=1
            )
        self.judge.set_lm(lm=dspy.LM(self.judge_agent, temperature=0.0, api_base=self.api_base))
        self.reflection = 2
        
    async def aforward(self, query):
        predictions = await asyncio.gather(*[self.base_blog.aforward(query=query) for _ in range(self.num_samples)])
        blog_ideas = [p.base for p in predictions]
        judge_score = self.judge(base=blog_ideas).rank 
        best_idea = judge_score.index(1)
        selected_idea = blog_ideas[best_idea]
        blog = None
        for _ in range(self.reflection):
            blog = self.adv_blog(base=selected_idea, draft=blog).content
        return blog
    
async def main():
    base_llms = ['ollama/qwen2.5:1.5b', 'ollama/qwen2.5:0.5b']
    adv_llms = ['ollama/qwen2.5:1.5b', 'ollama/qwen2.5:0.5b']
    temperature = [0.2,0.7,1.0]
    num_samples = [2,3]
    num_trials = 5
    results = []
    
    for i in range(num_trials):
        selected_base = random.choice(base_llms)
        selected_adv = random.choice(adv_llms)
        selected_temp1 = random.choice(temperature)
        selected_temp2 = random.choice(temperature)
        selected_num_samples = random.choice(num_samples)
        print(f"Trial {i+1}: Base LLM: {selected_base}, Adv LLM: {selected_adv}, Temp1: {selected_temp1}, Temp2: {selected_temp2}, Num Samples: {selected_num_samples}")
        evaluator = ConditionalEvaluator(agent1=selected_base, agent2=selected_adv, temp1=selected_temp1, temp2=selected_temp2, num_samples=selected_num_samples)
        start_time = time.perf_counter()
        # with mlflow.start_run(nested=True):
        #     mlflow.log_params({
        #         "base_llm": selected_base,
        #         "adv_llm": selected_adv,
        #         "temp1": selected_temp1,
        #         "temp2": selected_temp2,
        #         "num_samples": selected_num_samples,
        #         "trial": i + 1,
        #     })
        try:
            blog = await evaluator.aforward(query="Write a blog post about the benefits of meditation.")
            latency = time.perf_counter() - start_time
            # mlflow.log_metrics({"latency": latency})
            results.append({
                    "base_llm": selected_base,
                    "adv_llm": selected_adv,
                    "temp1": selected_temp1,
                    "temp2": selected_temp2,
                    "num_samples": selected_num_samples,
                    "latency": latency,
                    "blog": blog
                })
            print(f"Generated Blog:\n{blog}\n")
        except Exception as e:
                # mlflow.log_metric("error", 1)
            print(f"Error during evaluation: {e}")
    
    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)
    print("Evaluation completed. Results saved to evaluation_results.csv")

if __name__ == "__main__":
    asyncio.run(main())


