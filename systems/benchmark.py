from __future__ import annotations

import argparse
import json
import logging
import timeit
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

import basics
from basics.model import BasicsTransformerLM

from .log_utils import dump_config, make_run_dir, setup_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "small": ModelSpec(d_model=512, d_ff=2048, num_layers=8, num_heads=8),
    "medium": ModelSpec(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "large": ModelSpec(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    model_size: str
    context_length: int = 128
    batch_size: int = 4
    vocab_size: int = 10_000
    warmup_steps: int = 5
    measure_steps: int = 10
    mode: Literal["forward", "forward-backward", "train-step"] = "forward"
    use_bf16: bool = False
    use_memory_profiler: bool = False
    compile_model: bool = False
    nvtx: bool = False
    rope_theta: float = 10_000.0
    output_dir: Path | None = None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark and profile the Basics transformer.")
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "forward-backward", "train-step"], default="forward")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--use-memory-profiler", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--nvtx", action="store_true", help="Annotate steps with NVTX ranges.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override run dir. Defaults to logs/systems/<ts>_benchmark_<size>_<mode>/.",
    )
    return parser


def build_model(config: BenchmarkConfig) -> torch.nn.Module:
    spec = MODEL_SPECS[config.model_size]
    return BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=spec.d_model,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        d_ff=spec.d_ff,
        rope_theta=config.rope_theta,
    )


def make_random_batch(config: BenchmarkConfig, device: torch.device) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        device=device,
    )


def make_autocast_context(use_bf16: bool):
    if use_bf16 and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def run_single_step(
    model: torch.nn.Module,
    batch: torch.Tensor,
    mode: Literal["forward", "forward-backward", "train-step"],
    autocast_context,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    nvtx = _nvtx_range

    with nvtx("forward"):
        with autocast_context:
            logits = model(batch)
            if mode != "forward":
                loss = logits.float().mean()

    if mode in {"forward-backward", "train-step"}:
        with nvtx("backward"):
            loss.backward()

    if mode == "train-step":
        assert optimizer is not None
        with nvtx("optimizer"):
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _nvtx_range(name: str):
    if torch.cuda.is_available():
        try:
            import torch.cuda.nvtx as nvtx

            class _Ctx:
                def __enter__(self_inner):
                    nvtx.range_push(name)
                    return self_inner

                def __exit__(self_inner, *args):
                    nvtx.range_pop()

            return _Ctx()
        except ImportError:
            pass
    return nullcontext()


def maybe_start_memory_history(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)


def maybe_dump_memory_snapshot(enabled: bool, output_path: Path) -> None:
    if enabled and torch.cuda.is_available():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(output_path))
        torch.cuda.memory._record_memory_history(enabled=None)


def benchmark_model(config: BenchmarkConfig) -> dict[str, float]:
    run_dir = config.output_dir or make_run_dir(
        "systems", f"benchmark_{config.model_size}_{config.mode}"
    )
    setup_logging(run_dir)
    dump_config(run_dir, config)
    logger.info("Run dir: %s", run_dir)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    if config.compile_model:
        logger.info("Compiling model with torch.compile")
        model = torch.compile(model)
    optimizer = None
    if config.mode == "train-step":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch = make_random_batch(config, device)
    autocast = make_autocast_context(config.use_bf16)

    logger.info("Warmup: %d steps", config.warmup_steps)
    for _ in range(config.warmup_steps):
        run_single_step(model, batch, config.mode, autocast, optimizer)

    maybe_start_memory_history(config.use_memory_profiler)

    logger.info("Measuring: %d steps", config.measure_steps)
    times: list[float] = []
    for _ in range(config.measure_steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = timeit.default_timer()
        run_single_step(model, batch, config.mode, autocast, optimizer)
        t1 = timeit.default_timer()
        times.append(t1 - t0)

    if config.use_memory_profiler:
        snap_path = run_dir / "memory.pickle"
        maybe_dump_memory_snapshot(True, snap_path)
        logger.info("Memory snapshot: %s", snap_path)

    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / len(times)
    stddev = var ** 0.5
    result = {
        "mean_sec": mean,
        "stddev_sec": stddev,
        "min_sec": min(times),
        "max_sec": max(times),
        "config": {
            "model_size": config.model_size,
            "context_length": config.context_length,
            "batch_size": config.batch_size,
            "mode": config.mode,
            "use_bf16": config.use_bf16,
            "compile_model": config.compile_model,
        },
    }

    out_path = run_dir / "results.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Results: %s", json.dumps(result))
    return result


def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    """NVTX-annotated variant of basics.model.scaled_dot_product_attention."""
    import math

    import torch.cuda.nvtx as nvtx
    from einops import einsum

    from basics.nn_utils import softmax

    with nvtx.range("scaled_dot_product_attention"):
        d_k = K.shape[-1]
        with nvtx.range("attention_scores"):
            scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(d_k)
            if mask is not None:
                scores = torch.where(mask, scores, float("-inf"))
        with nvtx.range("softmax"):
            weights = softmax(scores, dim=-1)
        with nvtx.range("final_matmul"):
            out = einsum(weights, V, "... q k, ... k d -> ... q d")
    return out


def main() -> None:
    args = build_argparser().parse_args()
    config = BenchmarkConfig(
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        mode=args.mode,
        use_bf16=args.use_bf16,
        use_memory_profiler=args.use_memory_profiler,
        compile_model=args.compile_model,
        nvtx=args.nvtx,
        output_dir=args.output_dir,
    )
    if args.nvtx:
        import basics.model as bm

        bm.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    benchmark_model(config)


if __name__ == "__main__":
    main()
