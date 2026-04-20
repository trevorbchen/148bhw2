from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .log_utils import dump_config, make_run_dir, setup_logging

logger = logging.getLogger(__name__)


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, Tensor]:
    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_ids = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]
    full = [list(p) + list(o) for p, o in zip(prompt_ids, output_ids)]
    seq_lens = [len(f) - 1 for f in full]
    max_len = max(seq_lens)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    input_ids, labels, response_mask = [], [], []
    for p, o, f in zip(prompt_ids, output_ids, full):
        sl = len(f) - 1
        pad = [pad_id] * (max_len - sl)
        input_ids.append(list(f[:-1]) + pad)
        labels.append(list(f[1:]) + pad)
        p_len, o_len = len(p), len(o)
        mask = [False] * (p_len - 1) + [True] * o_len + [False] * (max_len - sl)
        response_mask.append(mask)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
    }


def compute_entropy(logits: Tensor) -> Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    result: dict[str, Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    masked = tensor * mask.to(tensor.dtype)
    summed = masked.sum() if dim is None else masked.sum(dim=dim)
    return summed / normalize_constant


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    rewards, fmt_rewards, ans_rewards = [], [], []
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        info = reward_fn(resp, gt)
        rewards.append(float(info["reward"]))
        fmt_rewards.append(float(info.get("format_reward", 0.0)))
        ans_rewards.append(float(info.get("answer_reward", 0.0)))

    raw = torch.tensor(rewards, dtype=torch.float32)
    grouped = raw.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    centered = grouped - mean
    if normalize_by_std:
        std = grouped.std(dim=1, keepdim=True, unbiased=False)
        normalized = centered / (std + advantage_eps)
    else:
        normalized = centered
    advantages = normalized.reshape(-1)

    metadata: dict[str, float] = {
        "reward_mean": float(raw.mean()),
        "reward_std": float(raw.std(unbiased=False)) if raw.numel() > 1 else 0.0,
        "reward_max": float(raw.max()),
        "reward_min": float(raw.min()),
        "format_reward_mean": float(sum(fmt_rewards) / len(fmt_rewards)) if fmt_rewards else 0.0,
        "answer_reward_mean": float(sum(ans_rewards) / len(ans_rewards)) if ans_rewards else 0.0,
        "group_reward_std_mean": float(grouped.std(dim=1, unbiased=False).mean()),
    }
    return advantages, raw, metadata


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    ratios = torch.exp(policy_log_probs - old_log_probs)
    clipped = torch.clamp(ratios, 1.0 - cliprange, 1.0 + cliprange)
    adv = advantages.expand_as(policy_log_probs)
    unclipped_obj = ratios * adv
    clipped_obj = clipped * adv
    loss = -torch.minimum(unclipped_obj, clipped_obj)
    was_clipped = (clipped_obj < unclipped_obj).to(policy_log_probs.dtype)
    metadata = {"clip_fraction": was_clipped.mean().detach()}
    return loss, metadata


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    per_token_loss, metadata = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    mask_f = response_mask.to(per_token_loss.dtype)
    masked = per_token_loss * mask_f
    per_example = masked.sum(dim=1) / mask_f.sum(dim=1)
    loss = per_example.mean() / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), metadata


def log_generations(
    prompts: Sequence[str],
    responses: Sequence[str],
    ground_truths: Sequence[str],
    reward_infos: Sequence[dict[str, float]],
    token_entropies: Sequence[float] | None = None,
    response_lengths: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for i, (p, r, gt, info) in enumerate(zip(prompts, responses, ground_truths, reward_infos)):
        entry: dict[str, Any] = {
            "prompt": p,
            "response": r,
            "ground_truth": gt,
            "format_reward": float(info.get("format_reward", 0.0)),
            "answer_reward": float(info.get("answer_reward", 0.0)),
            "reward": float(info.get("reward", 0.0)),
        }
        if token_entropies is not None:
            entry["avg_token_entropy"] = float(token_entropies[i])
        if response_lengths is not None:
            entry["response_length"] = int(response_lengths[i])
        entries.append(entry)

    if response_lengths is not None and reward_infos:
        lengths = [int(x) for x in response_lengths]
        correct_lens = [ln for ln, info in zip(lengths, reward_infos) if float(info.get("answer_reward", 0.0)) > 0.5]
        wrong_lens = [ln for ln, info in zip(lengths, reward_infos) if float(info.get("answer_reward", 0.0)) <= 0.5]
        summary = {
            "_summary": True,
            "avg_length": sum(lengths) / len(lengths) if lengths else 0.0,
            "avg_length_correct": sum(correct_lens) / len(correct_lens) if correct_lens else 0.0,
            "avg_length_incorrect": sum(wrong_lens) / len(wrong_lens) if wrong_lens else 0.0,
        }
        entries.append(summary)
    return entries


# --------------------------------------------------------------------------- #
# GRPO training loop (Section 3.5)
# --------------------------------------------------------------------------- #


def _flatten_rollouts(prompts: list[str], ground_truths: list[str], group_size: int) -> tuple[list[str], list[str]]:
    repeated_prompts = [p for p in prompts for _ in range(group_size)]
    repeated_gts = [gt for gt in ground_truths for _ in range(group_size)]
    return repeated_prompts, repeated_gts


def _compute_log_probs_over_microbatches(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    micro_batch_size: int,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    total = input_ids.shape[0]
    log_probs_chunks, entropy_chunks = [], []
    for start in range(0, total, micro_batch_size):
        end = start + micro_batch_size
        out = get_response_log_probs(
            model=model,
            input_ids=input_ids[start:end],
            labels=labels[start:end],
            return_token_entropy=return_token_entropy,
        )
        log_probs_chunks.append(out["log_probs"])
        if return_token_entropy:
            entropy_chunks.append(out["token_entropy"])
    result = {"log_probs": torch.cat(log_probs_chunks, dim=0)}
    if return_token_entropy:
        result["token_entropy"] = torch.cat(entropy_chunks, dim=0)
    return result


def train_grpo(
    *,
    policy_model,
    tokenizer,
    vllm_model,
    prompt_template: str,
    train_examples: Sequence[dict[str, Any]],
    val_examples: Sequence[dict[str, Any]],
    reward_fn: Callable[[str, str], dict[str, float]],
    output_dir: Path | None = None,
    run_name: str = "grpo",
    n_grpo_steps: int = 50,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 32,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 256,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 32,
    gradient_accumulation_steps: int = 16,
    cliprange: float = 1.0,
    normalize_by_std: bool = True,
    max_grad_norm: float = 1.0,
    eval_every: int = 5,
    n_eval_examples: int = 256,
    device: str | torch.device = "cuda",
    seed: int = 0,
    use_wandb: bool = False,
) -> dict[str, Any]:
    """Run the GRPO training loop. Requires vLLM for rollouts, HF model for training."""
    from vllm import SamplingParams

    assert train_batch_size % gradient_accumulation_steps == 0
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    output_dir = Path(output_dir) if output_dir is not None else make_run_dir("alignment", run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    dump_config(output_dir, {
        "n_grpo_steps": n_grpo_steps,
        "learning_rate": learning_rate,
        "advantage_eps": advantage_eps,
        "rollout_batch_size": rollout_batch_size,
        "group_size": group_size,
        "sampling_temperature": sampling_temperature,
        "sampling_min_tokens": sampling_min_tokens,
        "sampling_max_tokens": sampling_max_tokens,
        "epochs_per_rollout_batch": epochs_per_rollout_batch,
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "cliprange": cliprange,
        "normalize_by_std": normalize_by_std,
        "max_grad_norm": max_grad_norm,
        "eval_every": eval_every,
        "n_eval_examples": n_eval_examples,
        "device": str(device),
        "seed": seed,
        "n_train_examples": len(train_examples),
        "n_val_examples": len(val_examples),
    })
    logger.info("Run dir: %s", output_dir)
    log_path = output_dir / "train_log.jsonl"
    generations_path = output_dir / "generations.jsonl"

    device = torch.device(device)
    policy_model.to(device)

    optimizer = torch.optim.Adam(
        policy_model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=group_size,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=seed,
    )

    rng = torch.Generator().manual_seed(seed)
    train_idx_cursor = 0
    train_perm = torch.randperm(len(train_examples), generator=rng).tolist()

    def _next_prompt_batch(n: int) -> list[dict[str, Any]]:
        nonlocal train_idx_cursor, train_perm
        batch: list[dict[str, Any]] = []
        while len(batch) < n:
            if train_idx_cursor >= len(train_perm):
                train_perm = torch.randperm(len(train_examples), generator=rng).tolist()
                train_idx_cursor = 0
            batch.append(train_examples[train_perm[train_idx_cursor]])
            train_idx_cursor += 1
        return batch

    history: list[dict[str, Any]] = []
    gen_fp = generations_path.open("w", encoding="utf-8")
    log_fp = log_path.open("w", encoding="utf-8")

    try:
        for step in range(1, n_grpo_steps + 1):
            examples = _next_prompt_batch(n_prompts_per_rollout_batch)
            prompt_texts = [prompt_template.format(question=ex["question"]) for ex in examples]
            ground_truths = [str(ex["answer"]) for ex in examples]

            _sync_policy_to_vllm(policy_model, vllm_model)
            rollout_outputs = vllm_model.generate(prompt_texts, sampling_params, use_tqdm=False)

            rollout_responses: list[str] = []
            rollout_prompts: list[str] = []
            repeated_gts: list[str] = []
            for ex_idx, out in enumerate(rollout_outputs):
                for completion in out.outputs[:group_size]:
                    rollout_responses.append(completion.text)
                    rollout_prompts.append(prompt_texts[ex_idx])
                    repeated_gts.append(ground_truths[ex_idx])

            advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
                reward_fn=reward_fn,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_gts,
                group_size=group_size,
                advantage_eps=advantage_eps,
                normalize_by_std=normalize_by_std,
            )

            tok = tokenize_prompt_and_output(rollout_prompts, rollout_responses, tokenizer)
            input_ids = tok["input_ids"].to(device)
            labels = tok["labels"].to(device)
            response_mask = tok["response_mask"].to(device)

            policy_model.eval()
            with torch.no_grad():
                old = _compute_log_probs_over_microbatches(
                    policy_model, input_ids, labels, micro_train_batch_size
                )
            old_log_probs = old["log_probs"].detach()

            advantages_dev = advantages.to(device).unsqueeze(1)

            policy_model.train()
            step_losses: list[float] = []
            step_clip_fracs: list[float] = []
            step_entropies: list[float] = []

            perm = torch.randperm(rollout_batch_size, generator=rng).tolist()
            for epoch in range(epochs_per_rollout_batch):
                optimizer.zero_grad()
                for micro_start in range(0, rollout_batch_size, micro_train_batch_size):
                    micro_idx = perm[micro_start : micro_start + micro_train_batch_size]
                    idx_tensor = torch.tensor(micro_idx, dtype=torch.long, device=device)
                    mb_input = input_ids.index_select(0, idx_tensor)
                    mb_labels = labels.index_select(0, idx_tensor)
                    mb_mask = response_mask.index_select(0, idx_tensor)
                    mb_old = old_log_probs.index_select(0, idx_tensor)
                    mb_adv = advantages_dev.index_select(0, idx_tensor)

                    out = get_response_log_probs(
                        policy_model, mb_input, mb_labels, return_token_entropy=True
                    )
                    policy_log_probs = out["log_probs"]
                    token_entropy = out["token_entropy"].detach()

                    loss, meta = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=mb_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        advantages=mb_adv,
                        old_log_probs=mb_old,
                        cliprange=cliprange,
                    )
                    step_losses.append(float(loss.item()))
                    step_clip_fracs.append(float(meta["clip_fraction"].item()))
                    # Mean entropy only over response tokens
                    mb_mask_f = mb_mask.to(token_entropy.dtype)
                    denom = mb_mask_f.sum().clamp_min(1)
                    step_entropies.append(float((token_entropy * mb_mask_f).sum().item() / denom.item()))

                grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
                optimizer.step()

            log_row: dict[str, Any] = {
                "step": step,
                "loss": sum(step_losses) / max(len(step_losses), 1),
                "clip_fraction": sum(step_clip_fracs) / max(len(step_clip_fracs), 1),
                "response_entropy": sum(step_entropies) / max(len(step_entropies), 1),
                "grad_norm": float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm),
                **{f"reward/{k}": v for k, v in reward_meta.items()},
            }

            if step % eval_every == 0 or step == n_grpo_steps:
                val_reward = _evaluate_policy(
                    vllm_model=vllm_model,
                    reward_fn=reward_fn,
                    prompt_template=prompt_template,
                    examples=val_examples[:n_eval_examples],
                    sampling_params=SamplingParams(
                        temperature=sampling_temperature,
                        top_p=1.0,
                        max_tokens=sampling_max_tokens,
                        n=1,
                        stop=["</answer>"],
                        include_stop_str_in_output=True,
                        seed=seed,
                    ),
                )
                log_row["val/answer_reward"] = val_reward["answer_reward_mean"]
                log_row["val/format_reward"] = val_reward["format_reward_mean"]
                log_row["val/reward"] = val_reward["reward_mean"]

            history.append(log_row)
            log_fp.write(json.dumps(log_row) + "\n")
            log_fp.flush()
            logger.info(
                "step %d/%d loss=%.4f clip=%.3f ent=%.3f reward_mean=%.3f%s",
                step,
                n_grpo_steps,
                log_row["loss"],
                log_row["clip_fraction"],
                log_row["response_entropy"],
                log_row.get("reward/reward_mean", 0.0),
                f" val_reward={log_row.get('val/reward', float('nan')):.3f}" if "val/reward" in log_row else "",
            )

            # Sample a few generations to log
            for i in range(min(4, len(rollout_prompts))):
                info = reward_fn(rollout_responses[i], repeated_gts[i])
                gen_fp.write(json.dumps({
                    "step": step,
                    "prompt": rollout_prompts[i],
                    "response": rollout_responses[i],
                    "ground_truth": repeated_gts[i],
                    "reward": info.get("reward", 0.0),
                    "format_reward": info.get("format_reward", 0.0),
                    "answer_reward": info.get("answer_reward", 0.0),
                }) + "\n")
            gen_fp.flush()

            if use_wandb:
                try:
                    import wandb

                    wandb.log(log_row, step=step)
                except Exception:
                    pass
    finally:
        log_fp.close()
        gen_fp.close()

    final_dir = output_dir / "final"
    policy_model.save_pretrained(save_directory=str(final_dir))
    tokenizer.save_pretrained(save_directory=str(final_dir))
    logger.info("Saved final policy to %s", final_dir)
    return {"history": history, "output_dir": str(output_dir)}


def _sync_policy_to_vllm(policy_model, vllm_model) -> None:
    """Push current policy weights into the vLLM engine for on-policy rollouts."""
    try:
        state_dict = policy_model.state_dict()
        llm_model = vllm_model.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(state_dict.items())
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to sync policy weights into vLLM engine. "
            "Ensure vllm and the policy model share the same architecture."
        ) from exc


def _evaluate_policy(
    *,
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompt_template: str,
    examples: Sequence[dict[str, Any]],
    sampling_params,
) -> dict[str, float]:
    prompts = [prompt_template.format(question=ex["question"]) for ex in examples]
    gts = [str(ex["answer"]) for ex in examples]
    outputs = vllm_model.generate(prompts, sampling_params, use_tqdm=False)
    rewards, fmt, ans = [], [], []
    for out, gt in zip(outputs, gts):
        text = out.outputs[0].text
        info = reward_fn(text, gt)
        rewards.append(float(info["reward"]))
        fmt.append(float(info.get("format_reward", 0.0)))
        ans.append(float(info.get("answer_reward", 0.0)))
    n = max(len(rewards), 1)
    return {
        "reward_mean": sum(rewards) / n,
        "format_reward_mean": sum(fmt) / n,
        "answer_reward_mean": sum(ans) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRPO training on GSM8K.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--prompt", choices=["cot", "direct"], default="cot")
    parser.add_argument("--n-grpo-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    parser.add_argument("--cliprange", type=float, default=1.0)
    parser.add_argument("--no-normalize-by-std", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-min-tokens", type=int, default=4)
    parser.add_argument("--sampling-max-tokens", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--n-eval-examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="grpo")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM

    from .eval import load_gsm8k_examples
    from .prompts import COT_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE
    from .rewards import answer_tag_reward_fn

    train_examples = load_gsm8k_examples("train")
    val_examples = load_gsm8k_examples("test")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    vllm_model = LLM(model=args.model_name, dtype="bfloat16", gpu_memory_utilization=0.45, max_model_len=2048)

    template = str(COT_PROMPT_TEMPLATE) if args.prompt == "cot" else DIRECT_PROMPT_TEMPLATE

    train_grpo(
        policy_model=policy_model,
        tokenizer=tokenizer,
        vllm_model=vllm_model,
        prompt_template=template,
        train_examples=train_examples,
        val_examples=val_examples,
        reward_fn=answer_tag_reward_fn,
        output_dir=args.output_dir,
        run_name=args.run_name,
        n_grpo_steps=args.n_grpo_steps,
        learning_rate=args.learning_rate,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        sampling_temperature=args.sampling_temperature,
        sampling_min_tokens=args.sampling_min_tokens,
        sampling_max_tokens=args.sampling_max_tokens,
        epochs_per_rollout_batch=args.epochs_per_rollout_batch,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        cliprange=args.cliprange,
        normalize_by_std=not args.no_normalize_by_std,
        max_grad_norm=args.max_grad_norm,
        eval_every=args.eval_every,
        n_eval_examples=args.n_eval_examples,
        device=args.device,
        seed=args.seed,
        use_wandb=args.use_wandb,
    )


if __name__ == "__main__":
    main()
