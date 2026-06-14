import dspy
import pandas as pd
import random
import asyncio
import os

API = os.getenv("ELO_API")  # Retrieve the ELO API key from environment variables

dspy.configure(lm=dspy.LM("openrouter/deepseek/deepseek-r1", api_key=API), track_usage=True)
dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
)

class Compare(dspy.Signature):
    """
    You are a judge that compares two pieces of content and determines which one is better based on quality, coherence, and relevance.
    """
    content1: str = dspy.InputField(description="The first piece of content to compare")
    content2: str = dspy.InputField(description="The second piece of content to compare")
    winner: int = dspy.OutputField(description="The winner of the comparison: 1 for content1, 2 for content2, 0 for tie",le=1,ge=0)


comparer = dspy.ChainOfThought(Compare)

async def compare_contents(content1, content2):
    result = await comparer.acall(content1=content1, content2=content2)
    return result.winner

async def elo_test(data):
    idx_rang = [_ for _ in range(len(data))]
    picked = [0 for _ in range(len(data))]
    won = [0 for _ in range(len(data))]
    
    num_contests = 15
    calls = []
    pairs = []

    for _ in range(num_contests):
        picked_idx = random.sample(idx_rang, 2)
        pairs.append(picked_idx)
        content1 = data.iloc[picked_idx[0]]['blog']
        content2 = data.iloc[picked_idx[1]]['blog']
        winner = compare_contents(content1, content2)
        calls.append(winner)

    winners = await asyncio.gather(*calls)
    for p,w in zip(pairs, winners):
        picked[p[0]] += 1
        picked[p[1]] += 1
        if w == 1:
            won[p[0]] += 1
        elif w == 2:
            won[p[1]] += 1

    data['picked'] = picked
    data['won'] = won
    return data

if __name__ == "__main__":
    data = pd.read_csv('src/ContextEng/evaluation_results.csv')
    annotated_data = asyncio.run(elo_test(data))
    annotated_data.to_csv('src/ContextEng/elo_results.csv', index=False)