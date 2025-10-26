import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import sys

import nltk
import numpy as np
import transformers

from bert_score import score as bert_score
from dotenv import load_dotenv
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from openai import OpenAI
from pydantic import BaseModel, Field
from rouge_score import rouge_scorer
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.CRITICAL)
transformers.logging.set_verbosity_error()

# Download necessary NLTK resources
try:
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    print("NLTK resources downloaded successfully.")
except Exception as e:
    print(f"Warning: Failed to download NLTK resources: {e}")

try:
    sentence_model_name = "Qwen/Qwen3-Embedding-0.6B"
    sentence_model = SentenceTransformer(sentence_model_name)
    print(f"SentenceTransformer model : {sentence_model_name} loaded successfully.")
except Exception as e:
    print(f"Failed to load SentenceTransformer model: {e}")
    sentence_model = None


class LLMGrade(BaseModel):
    llm_judgment: str = Field(description="CORRECT or WRONG")
    llm_reasoning: str = Field(description="Explain why the answer is correct or incorrect.")


def calculate_rouge_scores(golden_answer, response):
    metrics = {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = scorer.score(golden_answer, response)
        metrics["rouge1_f"] = rouge_scores["rouge1"].fmeasure
        metrics["rouge2_f"] = rouge_scores["rouge2"].fmeasure
        metrics["rougeL_f"] = rouge_scores["rougeL"].fmeasure
    except Exception as e:
        print(f"Failed to calculate ROUGE scores: {e}")
    return metrics


def calculate_bleu_scores(gold_tokens, response_tokens):
    metrics = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    try:
        smoothing = SmoothingFunction().method1
        weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]

        for i, weight in enumerate(weights, 1):
            metrics[f"bleu{i}"] = sentence_bleu(
                [gold_tokens], response_tokens, weights=weight, smoothing_function=smoothing
            )
    except ZeroDivisionError:
        pass
    except Exception as e:
        print(f"Failed to calculate BLEU scores: {e}")

    return metrics


def calculate_meteor_score(gold_tokens, response_tokens):
    try:
        return meteor_score([gold_tokens], response_tokens)
    except Exception as e:
        print(f"Failed to calculate METEOR score: {e}")
        return 0.0


def calculate_semantic_similarity(golden_answer, response):
    global sentence_model

    try:
        if sentence_model is None:
            sentence_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

        gold_embedding = sentence_model.encode([golden_answer], show_progress_bar=False)[0]
        response_embedding = sentence_model.encode([response], show_progress_bar=False)[0]
        return 1 - cosine(gold_embedding, response_embedding)
    except Exception as e:
        print(f"Failed to calculate semantic similarity: {e}")
        return 0.0


def calculate_f1_score(gold_tokens, response_tokens):
    try:
        gold_set = set(gold_tokens)
        response_set = set(response_tokens)

        if len(gold_set) == 0 or len(response_set) == 0:
            return 0.0

        precision = len(gold_set.intersection(response_set)) / len(response_set)
        recall = len(gold_set.intersection(response_set)) / len(gold_set)

        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0
    except Exception as e:
        print(f"Failed to calculate F1 score: {e}")
        return 0.0


def calculate_nlp_metrics(golden_answer, response, context_token, options=None):
    if options is None:
        options = ["lexical", "semantic"]

    golden_answer = str(golden_answer) if golden_answer is not None else ""
    response = str(response) if response is not None else ""

    metrics = {"context_tokens": context_token}
    if "lexical" in options:
        gold_tokens = nltk.word_tokenize(golden_answer.lower())
        response_tokens = nltk.word_tokenize(response.lower())

        metrics["lexical"] = {}
        metrics["lexical"]["f1"] = calculate_f1_score(gold_tokens, response_tokens)
        metrics["lexical"].update(calculate_rouge_scores(golden_answer, response))
        metrics["lexical"].update(calculate_bleu_scores(gold_tokens, response_tokens))
        metrics["lexical"]["meteor"] = calculate_meteor_score(gold_tokens, response_tokens)

    if "semantic" in options:
        metrics["semantic"] = {}
        metrics["semantic"]["similarity"] = calculate_semantic_similarity(golden_answer, response)
        _, _, f1 = bert_score(
            [golden_answer], [response], lang="en", rescale_with_baseline=True, verbose=False
        )
        metrics["semantic"]["bert_f1"] = f1.item() if f1 is not None else 0.0

    return metrics

def lme_judge_model_template(question, golden_answer, response): 
    return f"""
    Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a ’gold’ (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Where did I buy my new tennis racket from?
    Gold answer: the sports store downtown
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it’s time for the real question:
    Question: {question}
    Gold answer: {golden_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """


def lme_grader(llm_client, question, golden_answer, response):
    system_prompt = """You are an expert grader that determines if answers to questions match a gold standard answer"""
    judge_prompt = lme_judge_model_template(
        question=question, golden_answer=golden_answer, response=response
    )

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": judge_prompt},
        ],
        temperature=0,
    )


    message_content = response.choices[0].message.content
    label = json.loads(message_content)["label"]
    parsed = LLMGrade(llm_judgment=label, llm_reasoning="")

    return parsed.llm_judgment.strip().lower() == "correct"


async def process_qa(
    user_id, response_data, llm_client, num_runs: int, nlp_options=None, executor=None
):
    question = response_data.get("question")
    golden_answer = response_data.get("original_answer", "")
    response = response_data.get("system_answer", "")
    context_token = response_data.get("context_token", 0)

    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(executor, lme_grader, llm_client, question, golden_answer, response)
        for _ in range(num_runs)
    ]
    judgments = await asyncio.gather(*tasks)
    judgments_dict = {f"judgment_{i + 1}": j for i, j in enumerate(judgments)}

    nlp_metrics = calculate_nlp_metrics(
        golden_answer=golden_answer, response=response, context_token=context_token, options=nlp_options
    )

    print("\n" + "=" * 80)
    print(f"🔍 Processed User: \033[1m{user_id}\033[0m")
    print("-" * 80)
    print(f"❓ Question: \n   {question}")
    print("-" * 80)
    print(
        f"📖 Golden Answer: \n   {golden_answer[:150]}..."
        if len(str(golden_answer)) > 150
        else f"📖 Golden Answer: \n   {golden_answer}"
    )
    print("-" * 80)
    print(
        f"💬 LLM Response: \n   {response[:150]}..."
        if len(str(response)) > 150
        else f"💬 Answer: \n   {response}"
    )
    print("-" * 80)

    judgments_formatted = []
    for run, correct in judgments_dict.items():
        status = "\033[92m✓ CORRECT\033[0m" if correct else "\033[91m✗ WRONG\033[0m"
        judgments_formatted.append(f"{run}: {status}")

    print(f"⚖️  Judgments: \n   {', '.join(judgments_formatted)}")
    print("=" * 80)

    graded_response = {
        "user_id": user_id,
        "category": response_data.get("category"),
        "question": question,
        "question_date": response_data.get("question_date"),
        "golden_answer": response_data.get("original_answer"),
        "answer": response,
        "llm_judgments": judgments_dict,
        "nlp_metrics": nlp_metrics,
        "processing_durations_s": response_data.get("processing_time"),
        "response_duration_s": response_data.get("response_time"),
        "search_duration_s": response_data.get("retrieval_time"),
        "total_duration_s": response_data.get("total_processing_time"),
    }
    return graded_response


def convert_numpy_types(obj):
    if isinstance(obj, np.number):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj


def evaluate_accuracy(results, num_runs):
    run_scores = []
    evaluated_count = 0

    for i in range(1, num_runs + 1):
        judgment_key = f"judgment_{i}"
        correct, total = 0, 0
        for _, response in results.items():
            if judgment_key in response["llm_judgments"]:
                total += 1
                if response["llm_judgments"][judgment_key]:
                    correct += 1
        if total > 0:
            run_scores.append(correct / total)
            evaluated_count += total
    evaluated_count = evaluated_count // num_runs
    return run_scores, evaluated_count


async def main(frame, version, nlp_options, num_runs=3, num_workers=5):
    print(f"Starting evaluation for {frame} version {version}...")

    load_dotenv()
    oai_client = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))

    response_path = f"results/all_lme_results.json"
    judged_path = f"demos/all_lme_judged.json"

    with open(response_path) as file:
        lme_responses = json.load(file)

    lme_eval_results = {}
    error_count = 0

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    tasks = [
        process_qa(response_data.get("question_id", f"item_{i}"), response_data, oai_client, num_runs, nlp_options, executor)
        for i, response_data in enumerate(lme_responses)
    ]
    results = []
    pbar = tqdm(total=len(tasks), desc="Processing users")
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            user_id = result["user_id"]
            lme_eval_results[user_id] = result
            results.append(result)
        except Exception as exc:
            print(f"[ERROR] Processing user failed: {exc}")
            error_count += 1
        pbar.update(1)
    pbar.close()
    executor.shutdown()

    run_scores, evaluated_count = evaluate_accuracy(lme_eval_results, num_runs)

    print("\n" + "=" * 80)
    print("\033[1;36m📊 EVALUATION SUMMARY\033[0m".center(80))
    print("=" * 80)

    if evaluated_count > 0:
        print(
            f"📋 \033[1mEvaluated:\033[0m \033[93m{evaluated_count}\033[0m responses across \033[93m{num_runs}\033[0m runs"
        )
        print(
            f"🎯 \033[1mLLM-as-a-Judge Mean Accuracy:\033[0m \033[92m{np.mean(run_scores):.4f}\033[0m"
        )
        print(f"🔍 \033[1mStandard Deviation:\033[0m \033[93m{np.std(run_scores):.4f}\033[0m")

        run_scores_formatted = [f"\033[94m{round(s, 4):.4f}\033[0m" for s in run_scores]
        print(f"🔢 \033[1mIndividual run scores:\033[0m [{', '.join(run_scores_formatted)}]")
    else:
        print("\033[91m⚠️  No responses were evaluated. LLM-as-a-Judge score: N/A (0/0)\033[0m")

    if error_count > 0:
        print(f"\033[91m⚠️  Encountered {error_count} errors during processing\033[0m")

    print("-" * 80)

    # Convert and save results
    lme_eval_results = convert_numpy_types(lme_eval_results)
    with open(judged_path, "w") as file:
        json.dump(lme_eval_results, file, indent=4)

    print("\033[92m✅ Evaluation completed successfully!\033[0m")
    print(f"📁 Results saved to: \033[1;94m{judged_path}\033[0m")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM responses using LLM-as-a-Judge.")
    parser.add_argument(
        "--lib",
        type=str,
        choices=["memoryos", "mem0-local", "mem0-api", "memos-local", "zep", "memos-api", "zep", "memobase"],
    )
    parser.add_argument(
        "--version", type=str, default="v1", help="Version of the evaluation framework."
    )
    parser.add_argument(
        "--options",
        type=str,
        nargs="+",
        default=["lexical", "semantic"],
        choices=["lexical", "semantic"],
        help="NLP options to use for evaluation.",
    )
    parser.add_argument(
        "--num_runs", type=int, default=3, help="Number of runs for LLM-as-a-Judge evaluation."
    )
    parser.add_argument(
        "--workers", type=int, default=3, help="Number of runs for LLM-as-a-Judge evaluation."
    )

    args = parser.parse_args()
    asyncio.run(
        main(
            frame=args.lib,
            version=args.version,
            nlp_options=args.options,
            num_runs=args.num_runs,
            num_workers=args.workers,
        )
    )
