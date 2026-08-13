import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.agent import run_agent
from openai import AsyncOpenAI
from app.core.config import settings
from app.agent import initialize
import asyncio

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# Test dataset — 7 corpus questions, 3 web questions
TEST_CASES = [
    {
        "query": "What is retrieval augmented generation?",
        "expected_tool": "search_corpus"
    },
    {
        "query": "How does cross-encoder reranking improve retrieval quality?",
        "expected_tool": "search_corpus"
    },
    {
        "query": "What are the differences between RAG and fine-tuning?",
        "expected_tool": "search_corpus"
    },
    {
        "query": "Explain how attention mechanisms work in transformers",
        "expected_tool": "search_corpus"
    },
    {
        "query": "What is RLHF and how is it used to train language models?",
        "expected_tool": "search_corpus"
    },
    {
        "query": "What are the main approaches to knowledge graph reasoning?",
        "expected_tool": "search_corpus"
    },
    {
        "query": "How does LoRA fine-tuning work?",
        "expected_tool": "search_corpus"
    },
    {
        "query": "What did Anthropic announce this week?",
        "expected_tool": "search_web"
    },
    {
        "query": "What are the latest AI news today?",
        "expected_tool": "search_web"
    },
    {
        "query": "Who won the most recent AI safety conference?",
        "expected_tool": "search_web"
    }
]

async def score_faithfulness(query: str, context: str, answer: str) -> float:
    """Ask LLM to judge whether the answer is grounded in the context."""

    prompt = f"""You are an evaluation judge. Score the faithfulness of the answer below.
    Faithfulness means: does the answer only contain information that is present in the context?
    If the answer makes claims not supported by the context, it is not faithful.

    Question: {query}

    Context:
    {context}

    Answer:
    {answer}

    Score the faithfulness from 0.0 to 1.0 where:
    1.0 = answer is completely grounded in the context
    0.5 = answer is partially grounded, some unsupported claims
    0.0 = answer contains mostly unsupported claims

    Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    
    try:
         return float(response.choices[0].message.content.strip())
    except:
        return 0.0  # Return 0 if there's an error in parsing the score

async def score_relevancy(query: str, answer: str) -> float:
    """Ask LLM to judge whether the answer addresses the question."""

    prompt = f"""You are an evaluation judge. Score the relevancy of the answer below.
    Relevancy means: does the answer actually address what was asked?

    Question: {query}

    Answer:
    {answer}

    Score the relevancy from 0.0 to 1.0 where:
    1.0 = answer directly and completely addresses the question
    0.5 = answer partially addresses the question
    0.0 = answer does not address the question at all

    Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.0

async def evaluate():
    await initialize()  # Ensure the agent is initialized before evaluation
    results = []
    faithfulness_scores = []
    relevancy_scores = []
    tool_correct = 0

    print(f"Running evaluation on {len(TEST_CASES)} test cases...\n")

    for i, test in enumerate(TEST_CASES):
        print(f"[{i+1}/{len(TEST_CASES)}] {test['query'][:60]}...")

        try:
            result = await run_agent(test["query"])

            # tool selection accuracy
            tool_correct_flag = result["tool_used"] == test["expected_tool"]
            if tool_correct_flag:
                tool_correct += 1

            # faithfulness — only meaningful for corpus queries
            faith_score = await score_faithfulness(
                query=test["query"],
                context=result["context"],
                answer=result["answer"]
            )
            faithfulness_scores.append(faith_score)

            # answer relevancy
            rel_score = await score_relevancy(
                query=test["query"],
                answer=result["answer"]
            )
            relevancy_scores.append(rel_score)

            print(f"  Tool: {result['tool_used']} (expected: {test['expected_tool']}) {'✓' if tool_correct_flag else '✗'}")
            print(f"  Faithfulness: {faith_score:.2f}")
            print(f"  Relevancy: {rel_score:.2f}\n")

        except Exception as e:
            print(f"  ERROR: {e}\n")
            faithfulness_scores.append(0.0)
            relevancy_scores.append(0.0)

    # aggregate scores
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Faithfulness:          {sum(faithfulness_scores)/len(faithfulness_scores):.3f}")
    print(f"Answer Relevancy:      {sum(relevancy_scores)/len(relevancy_scores):.3f}")
    print(f"Tool Selection:        {tool_correct}/{len(TEST_CASES)} ({tool_correct/len(TEST_CASES)*100:.0f}%)")


if __name__ == "__main__":
    asyncio.run(evaluate())