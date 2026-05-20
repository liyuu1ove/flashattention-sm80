#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "build" / "bench_xydnn_vs_cudnn"


@dataclass
class Case:
    name: str
    env: dict[str, str]
    max_abs_tol: float = 5.0e-2
    rel_max_tol: float = 5.0e-2
    cudnn_max_abs_tol: float = 5.0e-2
    cudnn_rel_max_tol: float = 5.0e-2


@dataclass
class PrecisionMetrics:
    max_abs: float
    mean_abs: float
    rel_max: float


def run(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def parse_precision_line(output: str, label: str) -> PrecisionMetrics | None:
    match = re.search(
        rf"precision {re.escape(label)} "
        r"max_abs=([0-9.eE+-]+) "
        r"mean_abs=([0-9.eE+-]+) "
        r"rel_max=([0-9.eE+-]+)",
        output,
    )
    if match is None:
        return None
    return PrecisionMetrics(
        max_abs=float(match.group(1)),
        mean_abs=float(match.group(2)),
        rel_max=float(match.group(3)),
    )


def base_env() -> dict[str, str]:
    return {
        "BATCH": "2",
        "SEQLEN": "128",
        "HEADS": "4",
        "KV_HEADS": "4",
        "HEAD_DIM": "64",
        "WARMUP": os.environ.get("WARMUP", "2"),
        "ITERS": os.environ.get("ITERS", "5"),
    }


def make_case(name: str, **kwargs: object) -> Case:
    env = base_env()
    env.update({key: str(value) for key, value in kwargs.items()})
    return Case(name=name, env=env)


def smoke_cases() -> list[Case]:
    return [
        make_case("dense-fp16", DTYPE="fp16"),
        make_case("dense-bf16", DTYPE="bf16"),
        make_case("causal-fp16", DTYPE="fp16", CAUSAL=1),
        make_case("causal-bf16", DTYPE="bf16", CAUSAL=1),
        make_case("varlen-fp16", DTYPE="fp16", VARLEN=1),
        make_case("varlen-bf16", DTYPE="bf16", VARLEN=1),
        make_case("gqa-fp16", DTYPE="fp16", KV_HEADS=2),
        make_case("gqa-bf16", DTYPE="bf16", KV_HEADS=2),
        make_case("softcap-fp16", DTYPE="fp16", SOFTCAP=20),
        make_case("window-fp16", DTYPE="fp16", WINDOW_LEFT=32, WINDOW_RIGHT=16),
        make_case("dropout-fp16", DTYPE="fp16", DROPOUT=0.1, DROPOUT_SEED=1234),
        make_case("dropout-bf16", DTYPE="bf16", DROPOUT=0.1, DROPOUT_SEED=1234),
    ]


def full_cases() -> list[Case]:
    cases = smoke_cases()
    for dtype in ("fp16", "bf16"):
        for head_dim in (32, 96, 128, 192, 256):
            cases.append(make_case(f"dense-{dtype}-d{head_dim}", DTYPE=dtype, HEAD_DIM=head_dim))
            cases.append(make_case(f"causal-{dtype}-d{head_dim}", DTYPE=dtype, HEAD_DIM=head_dim, CAUSAL=1))
        for seqlen in (64, 128, 256):
            cases.append(make_case(f"dense-{dtype}-s{seqlen}", DTYPE=dtype, SEQLEN=seqlen))
            cases.append(make_case(f"causal-{dtype}-s{seqlen}", DTYPE=dtype, SEQLEN=seqlen, CAUSAL=1))
        cases.append(make_case(f"dense-{dtype}-b1-h1", DTYPE=dtype, BATCH=1, SEQLEN=96, HEADS=1, KV_HEADS=1))
        cases.append(make_case(f"mqa-{dtype}", DTYPE=dtype, HEADS=8, KV_HEADS=1))
        cases.append(make_case(f"varlen-gqa-{dtype}", DTYPE=dtype, HEADS=8, KV_HEADS=2, VARLEN=1))
        cases.append(make_case(f"window-{dtype}-s256", DTYPE=dtype, SEQLEN=256, WINDOW_LEFT=64, WINDOW_RIGHT=32))
        cases.append(make_case(f"softcap-varlen-{dtype}", DTYPE=dtype, VARLEN=1, SOFTCAP=30))
        cases.append(make_case(f"dropout-{dtype}-causal", DTYPE=dtype, CAUSAL=1, DROPOUT=0.1, DROPOUT_SEED=1234))
        cases.append(make_case(f"dropout-{dtype}-varlen", DTYPE=dtype, VARLEN=1, DROPOUT=0.1, DROPOUT_SEED=1234))
    return cases


def main() -> int:
    suite = os.environ.get("XYDNN_TEST_SUITE", "smoke")
    build = os.environ.get("XYDNN_SKIP_BUILD", "0") != "1"
    if suite not in {"smoke", "full"}:
        print(f"XYDNN_TEST_SUITE must be smoke or full, got {suite!r}", file=sys.stderr)
        return 2

    if build:
        print("[build] make bench-xydnn-vs-cudnn -j1", flush=True)
        build_result = run(["make", "bench-xydnn-vs-cudnn", "-j1"])
        print(build_result.stdout, end="")
        if build_result.returncode != 0:
            return build_result.returncode

    if not BENCH.exists():
        print(f"Missing {BENCH}; run make bench-xydnn-vs-cudnn first", file=sys.stderr)
        return 2

    cases = smoke_cases() if suite == "smoke" else full_cases()
    failed: list[str] = []
    skipped_cudnn = 0
    compared_cudnn = 0

    print(f"[matrix] suite={suite} cases={len(cases)}", flush=True)
    for index, case in enumerate(cases, 1):
        label = f"[{index:02d}/{len(cases):02d}] {case.name}"
        print(f"{label} ... ", end="", flush=True)
        result = run([str(BENCH)], case.env)
        output = result.stdout
        if result.returncode != 0:
            print("FAIL")
            print(output)
            failed.append(case.name)
            continue

        fa_metrics = parse_precision_line(output, "xyDNN-vs-FA")
        if fa_metrics is None:
            print("FAIL")
            print(output)
            failed.append(case.name)
            continue
        if fa_metrics.max_abs > case.max_abs_tol or fa_metrics.rel_max > case.rel_max_tol:
            print(f"FAIL fa_max_abs={fa_metrics.max_abs:.6g} fa_rel_max={fa_metrics.rel_max:.6g}")
            print(output)
            failed.append(case.name)
            continue

        if "real cuDNN backend skipped:" in output:
            skipped_cudnn += 1
            print(
                f"PASS fa_max_abs={fa_metrics.max_abs:.6g} "
                f"fa_rel_max={fa_metrics.rel_max:.6g} cudnn=skip"
            )
            continue

        cudnn_metrics = parse_precision_line(output, "xyDNN-vs-cuDNN")
        if cudnn_metrics is None:
            print("FAIL cudnn_metrics=missing")
            print(output)
            failed.append(case.name)
            continue
        if (cudnn_metrics.max_abs > case.cudnn_max_abs_tol or
                cudnn_metrics.rel_max > case.cudnn_rel_max_tol):
            print(
                f"FAIL cudnn_max_abs={cudnn_metrics.max_abs:.6g} "
                f"cudnn_rel_max={cudnn_metrics.rel_max:.6g}"
            )
            print(output)
            failed.append(case.name)
            continue

        compared_cudnn += 1
        print(
            f"PASS fa_max_abs={fa_metrics.max_abs:.6g} "
            f"fa_rel_max={fa_metrics.rel_max:.6g} "
            f"cudnn_max_abs={cudnn_metrics.max_abs:.6g} "
            f"cudnn_rel_max={cudnn_metrics.rel_max:.6g}"
        )

    passed = len(cases) - len(failed)
    print(
        f"[summary] passed={passed} failed={len(failed)} "
        f"cudnn_compared={compared_cudnn} cudnn_skipped={skipped_cudnn}"
    )
    if failed:
        print("[failed] " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
