from __future__ import annotations

import argparse
import json
import logging
import math
import timeit
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch

from basics.model import scaled_dot_product_attention

from .log_utils import dump_config, make_run_dir, setup_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    head_dims: tuple[int, ...] = (16, 32, 64, 128)
    sequence_lengths: tuple[int, ...] = (64, 128, 256, 512, 1024)
    batch_size: int = 8
    forward_passes: int = 100
    backward_passes: int = 100
    compile_attention: bool = False
    output_dir: Path | None = None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark attention implementations.")
    parser.add_argument("--compile-attention", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override run dir. Defaults to logs/systems/<ts>_attention_<eager|compiled>/.",
    )
    return parser


def iter_benchmark_shapes(config: AttentionBenchmarkConfig) -> Iterable[tuple[int, int]]:
    for head_dim in config.head_dims:
        for sequence_length in config.sequence_lengths:
            yield head_dim, sequence_length


def make_qkv(
    batch_size: int,
    sequence_length: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch_size, sequence_length, head_dim)
    q = torch.randn(shape, device=device, requires_grad=True)
    k = torch.randn(shape, device=device, requires_grad=True)
    v = torch.randn(shape, device=device, requires_grad=True)
    return q, k, v


def _causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    seq = torch.arange(sequence_length, device=device)
    return seq[:, None] >= seq[None, :]


def benchmark_attention_once(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    forward_passes: int,
    backward_passes: int,
    attention_fn,
) -> dict[str, float]:
    device = q.device
    mask = _causal_mask(q.shape[1], device)

    # Warmup
    out = attention_fn(q, k, v, mask)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    fwd_times: list[float] = []
    for _ in range(forward_passes):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = timeit.default_timer()
        out = attention_fn(q, k, v, mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fwd_times.append(timeit.default_timer() - t0)

    mem_before_backward = (
        torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    )

    bwd_times: list[float] = []
    for _ in range(backward_passes):
        out = attention_fn(q, k, v, mask)
        loss = out.sum()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = timeit.default_timer()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bwd_times.append(timeit.default_timer() - t0)
        q.grad = k.grad = v.grad = None

    return {
        "forward_mean_sec": sum(fwd_times) / len(fwd_times),
        "forward_stddev_sec": _stddev(fwd_times),
        "backward_mean_sec": sum(bwd_times) / len(bwd_times),
        "backward_stddev_sec": _stddev(bwd_times),
        "mem_before_backward_mb": mem_before_backward,
    }


def _stddev(xs: list[float]) -> float:
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def benchmark_attention_grid(config: AttentionBenchmarkConfig) -> list[dict[str, float | int | str]]:
    suffix = "compiled" if config.compile_attention else "eager"
    run_dir = config.output_dir or make_run_dir("systems", f"attention_{suffix}")
    setup_logging(run_dir)
    dump_config(run_dir, config)
    logger.info("Run dir: %s", run_dir)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention_fn = scaled_dot_product_attention
    if config.compile_attention:
        logger.info("Compiling attention with torch.compile")
        attention_fn = torch.compile(attention_fn)

    results: list[dict[str, float | int | str]] = []
    for head_dim, sequence_length in iter_benchmark_shapes(config):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            q, k, v = make_qkv(config.batch_size, sequence_length, head_dim, device)
            metrics = benchmark_attention_once(
                q, k, v, config.forward_passes, config.backward_passes, attention_fn
            )
            row = {
                "head_dim": head_dim,
                "sequence_length": sequence_length,
                "batch_size": config.batch_size,
                "compile_attention": config.compile_attention,
                **metrics,
            }
        except torch.cuda.OutOfMemoryError:
            row = {
                "head_dim": head_dim,
                "sequence_length": sequence_length,
                "batch_size": config.batch_size,
                "compile_attention": config.compile_attention,
                "error": "OOM",
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append(row)
        logger.info(json.dumps(row))

    out_path = run_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s (%d rows)", out_path, len(results))
    return results


def main() -> None:
    args = build_argparser().parse_args()
    config = AttentionBenchmarkConfig(
        compile_attention=args.compile_attention,
        output_dir=args.output_dir,
    )
    benchmark_attention_grid(config)


if __name__ == "__main__":
    main()
