from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .log_utils import dump_config, make_run_dir, setup_logging
from .prompts import COT_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE
from .rewards import answer_tag_reward_fn, extract_answer_from_tags, r1_zero_reward_fn

logger = logging.getLogger(__name__)


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_VALIDATION_SIZE = 256

REWARD_FNS: dict[str, Callable[[str, str], dict[str, float]]] = {
    "r1_zero": r1_zero_reward_fn,
    "answer_tag": answer_tag_reward_fn,
}


def _resolve_reward_fn(name: str | None, mode: str) -> Callable[[str, str], dict[str, float]]:
    """PDF p13: r1_zero is the GSM8K default. Direct prompt has no <think>, so we
    fall back to answer_tag for direct mode unless the user overrides explicitly."""
    if name is None or name == "auto":
        return answer_tag_reward_fn if mode == "direct" else r1_zero_reward_fn
    if name not in REWARD_FNS:
        raise ValueError(f"Unknown reward fn: {name}. Choose from {list(REWARD_FNS)}.")
    return REWARD_FNS[name]


def load_gsm8k_examples(split: str) -> list[dict[str, Any]]:
    """Load GSM8K examples. Prefers local parquet files, falls back to `datasets`."""
    repo_root = Path(__file__).resolve().parent.parent
    local = repo_root / "data" / "gsm8k" / f"{split}.parquet"
    if local.exists():
        import pandas as pd

        df = pd.read_parquet(local)
        records = df.to_dict(orient="records")
    else:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split=split)
        records = [dict(r) for r in ds]

    examples: list[dict[str, Any]] = []
    for r in records:
        question = r["question"]
        raw_answer = r["answer"]
        gold = _extract_gsm8k_answer(raw_answer)
        examples.append({"question": question, "answer": gold, "raw_answer": raw_answer})
    return examples


def _extract_gsm8k_answer(answer_field: str) -> str:
    """GSM8K answers end with '#### <final>'. Extract the final numeric answer."""
    if "####" in answer_field:
        return answer_field.rsplit("####", 1)[-1].strip().replace(",", "")
    return answer_field.strip()


def build_prompts(examples: Sequence[dict[str, Any]], prompt_template) -> list[str]:
    return [prompt_template.format(question=ex["question"]) for ex in examples]


def evaluate_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    eval_sampling_params,
    ground_truths: Sequence[str] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    outputs = vllm_model.generate(list(prompts), eval_sampling_params, use_tqdm=True)

    records: list[dict[str, Any]] = []
    reward_sum = fmt_sum = ans_sum = 0.0
    categories = Counter()

    for i, out in enumerate(outputs):
        gen = out.outputs[0].text
        gt = ground_truths[i] if ground_truths is not None else ""
        info = reward_fn(gen, gt) if ground_truths is not None else {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}
        reward_sum += info["reward"]
        fmt_sum += info["format_reward"]
        ans_sum += info["answer_reward"]
        cat = _category(info)
        categories[cat] += 1
        records.append({
            "prompt": prompts[i],
            "generated": gen,
            "ground_truth": gt,
            "format_reward": info["format_reward"],
            "answer_reward": info["answer_reward"],
            "reward": info["reward"],
            "category": cat,
        })

    n = max(len(records), 1)
    summary = {
        "n": len(records),
        "reward_mean": reward_sum / n,
        "format_reward_mean": fmt_sum / n,
        "answer_reward_mean": ans_sum / n,
        "categories": dict(categories),
    }

    if output_path is not None:
        write_evaluation_results({"summary": summary, "records": records}, Path(output_path))

    return {"summary": summary, "records": records}


def _category(info: dict[str, float]) -> str:
    f = info.get("format_reward", 0.0)
    a = info.get("answer_reward", 0.0)
    if f >= 0.5 and a >= 0.5:
        return "format1_answer1"
    if f >= 0.5 and a < 0.5:
        return "format1_answer0"
    return "format0_answer0"


def write_evaluation_results(results: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results["summary"], fh, indent=2)
        fh.write("\n")
    records_path = output_path.with_suffix(".jsonl")
    with records_path.open("w", encoding="utf-8") as fh:
        for rec in results["records"]:
            fh.write(json.dumps(rec) + "\n")


def _load_vllm(model_name: str):
    from vllm import LLM

    return LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048)


def _base_sampling_params(n: int = 1, max_tokens: int = 1024):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_tokens,
        n=n,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )


def _resolve_run_dir(output_dir: Path | None, run_name: str) -> Path:
    if output_dir is not None:
        run_dir = Path(output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    return make_run_dir("alignment", run_name)


def run_direct_baseline(
    output_dir: Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    split: str = "train",
    limit: int | None = None,
    reward_fn_name: str | None = None,
) -> dict[str, Any]:
    reward_fn = _resolve_reward_fn(reward_fn_name, "direct")
    run_dir = _resolve_run_dir(output_dir, f"eval_direct_{split}")
    setup_logging(run_dir)
    dump_config(run_dir, {
        "mode": "direct", "model_name": model_name, "split": split, "limit": limit,
        "reward_fn": reward_fn.__name__,
    })
    logger.info("Run dir: %s | reward_fn: %s", run_dir, reward_fn.__name__)

    examples = load_gsm8k_examples(split)
    if limit is not None:
        examples = examples[:limit]
    logger.info("Loaded %d %s examples", len(examples), split)
    prompts = build_prompts(examples, DIRECT_PROMPT_TEMPLATE)
    gts = [ex["answer"] for ex in examples]

    llm = _load_vllm(model_name)
    result = evaluate_vllm(
        vllm_model=llm,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_sampling_params=_base_sampling_params(),
        ground_truths=gts,
        output_path=run_dir / "summary.json",
    )
    logger.info("Summary: %s", json.dumps(result["summary"]))
    return result


def run_cot_baseline(
    output_dir: Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    split: str = "train",
    limit: int | None = None,
    reward_fn_name: str | None = None,
) -> dict[str, Any]:
    reward_fn = _resolve_reward_fn(reward_fn_name, "cot")
    run_dir = _resolve_run_dir(output_dir, f"eval_cot_{split}")
    setup_logging(run_dir)
    dump_config(run_dir, {
        "mode": "cot", "model_name": model_name, "split": split, "limit": limit,
        "reward_fn": reward_fn.__name__,
    })
    logger.info("Run dir: %s | reward_fn: %s", run_dir, reward_fn.__name__)

    examples = load_gsm8k_examples(split)
    if limit is not None:
        examples = examples[:limit]
    logger.info("Loaded %d %s examples", len(examples), split)
    prompts = build_prompts(examples, COT_PROMPT_TEMPLATE)
    gts = [ex["answer"] for ex in examples]

    llm = _load_vllm(model_name)
    result = evaluate_vllm(
        vllm_model=llm,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_sampling_params=_base_sampling_params(),
        ground_truths=gts,
        output_path=run_dir / "summary.json",
    )
    logger.info("Summary: %s", json.dumps(result["summary"]))
    return result


def run_self_consistency_baseline(
    output_dir: Path | None = None,
    k: int = 5,
    model_name: str = DEFAULT_MODEL_NAME,
    split: str = "train",
    limit: int | None = None,
    reward_fn_name: str | None = None,
) -> dict[str, Any]:
    from vllm import SamplingParams

    reward_fn = _resolve_reward_fn(reward_fn_name, "self_consistency")
    run_dir = _resolve_run_dir(output_dir, f"eval_selfcons_k{k}_{split}")
    setup_logging(run_dir)
    dump_config(run_dir, {
        "mode": "self_consistency", "k": k, "model_name": model_name, "split": split, "limit": limit,
        "reward_fn": reward_fn.__name__,
    })
    logger.info("Run dir: %s | reward_fn: %s", run_dir, reward_fn.__name__)

    examples = load_gsm8k_examples(split)
    if limit is not None:
        examples = examples[:limit]
    logger.info("Loaded %d %s examples", len(examples), split)
    prompts = build_prompts(examples, COT_PROMPT_TEMPLATE)
    gts = [ex["answer"] for ex in examples]

    llm = _load_vllm(model_name)
    sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        n=k,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, sampling, use_tqdm=True)

    records: list[dict[str, Any]] = []
    reward_sum = fmt_sum = ans_sum = 0.0
    tie_count = 0

    for i, out in enumerate(outputs):
        texts = [c.text for c in out.outputs]
        answers = [extract_answer_from_tags(t) for t in texts]
        parsed = [a for a in answers if a is not None]
        if parsed:
            counts = Counter(parsed).most_common()
            top_count = counts[0][1]
            majority = counts[0][0]
            if len(counts) > 1 and counts[1][1] == top_count:
                tie_count += 1
            # Include the </think> marker so r1_zero_reward_fn passes format.
            synthesized = f"</think> <answer>{majority}</answer>"
        else:
            synthesized = "</think> <answer></answer>"
        info = reward_fn(synthesized, gts[i])
        reward_sum += info["reward"]
        fmt_sum += info["format_reward"]
        ans_sum += info["answer_reward"]
        records.append({
            "prompt": prompts[i],
            "samples": texts,
            "parsed_answers": answers,
            "majority_answer": synthesized,
            "ground_truth": gts[i],
            "format_reward": info["format_reward"],
            "answer_reward": info["answer_reward"],
            "reward": info["reward"],
        })

    n = max(len(records), 1)
    summary = {
        "n": len(records),
        "k": k,
        "reward_mean": reward_sum / n,
        "format_reward_mean": fmt_sum / n,
        "answer_reward_mean": ans_sum / n,
        "tie_count": tie_count,
    }

    write_evaluation_results({"summary": summary, "records": records}, run_dir / "summary.json")
    logger.info("Summary: %s", json.dumps(summary))
    return {"summary": summary, "records": records}


def get_prompt_template(use_cot: bool):
    return COT_PROMPT_TEMPLATE if use_cot else DIRECT_PROMPT_TEMPLATE


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GSM8K zero-shot baselines with vLLM.")
    parser.add_argument("--mode", choices=["direct", "cot", "self_consistency"], default="direct")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--split", default="train", help="GSM8K split. PDF p13 asks for train.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=5, help="Number of samples for self-consistency.")
    parser.add_argument(
        "--reward-fn",
        choices=["auto", *REWARD_FNS],
        default="auto",
        help="Reward fn. 'auto' = answer_tag for direct, r1_zero for cot/self_consistency (PDF p13).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override run dir. Defaults to logs/alignment/<ts>_eval_<mode>_<split>/.",
    )
    args = parser.parse_args()

    if args.mode == "direct":
        run_direct_baseline(args.output_dir, args.model_name, args.split, args.limit, args.reward_fn)
    elif args.mode == "cot":
        run_cot_baseline(args.output_dir, args.model_name, args.split, args.limit, args.reward_fn)
    else:
        run_self_consistency_baseline(args.output_dir, args.k, args.model_name, args.split, args.limit, args.reward_fn)


if __name__ == "__main__":
    main()
