#!/usr/bin/env python3
"""Compare ONNX OCR output quality against examples/result official outputs."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in s.split("\n")).strip()


def relaxed_text(s: str) -> str:
    s = normalize_text(s).lower()
    s = s.replace("$$", " ")
    s = re.sub(r"[\s`#*_{}\\]", "", s)
    s = s.replace("，", ",").replace("。", ".")
    return s


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(b) > len(a):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_f1_no_ws(pred: str, ref: str) -> tuple[float, float, float]:
    pred_s = re.sub(r"\s+", "", normalize_text(pred))
    ref_s = re.sub(r"\s+", "", normalize_text(ref))
    cp = Counter(pred_s)
    cr = Counter(ref_s)
    inter = sum((cp & cr).values())
    p = inter / max(1, len(pred_s))
    r = inter / max(1, len(ref_s))
    f1 = 0.0 if (p + r) == 0 else (2 * p * r / (p + r))
    return f1, p, r


def infer_task(stem: str) -> str:
    s = stem.lower()
    if "table" in s:
        return "table"
    if "formula" in s:
        return "formula"
    if "seal" in s or "handwritten" in s or "code" in s:
        return "text"
    return "document"


def run_ocr(
    artifact_dir: Path,
    image_path: Path,
    out_path: Path,
    task: str,
    device: str,
    cuda_no_fallback: bool,
    disable_kv_iobinding: bool,
    vision_policy: str,
    disable_cuda_ep_tuning: bool,
) -> float:
    cmd = [
        sys.executable,
        "run_onnx_static.py",
        "--artifact_dir",
        str(artifact_dir),
        "--image",
        str(image_path),
        "--task",
        task,
        "--device",
        device,
        "--official_quality",
        "--out_text",
        str(out_path),
    ]
    if cuda_no_fallback:
        cmd.append("--cuda_no_fallback")
    if disable_kv_iobinding:
        cmd.append("--disable_kv_iobinding")
    if vision_policy:
        cmd.extend(["--vision_policy", vision_policy])
    if disable_cuda_ep_tuning:
        cmd.append("--disable_cuda_ep_tuning")
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", default="artifact_glm_ocr_web_split")
    ap.add_argument("--examples_root", default="examples")
    ap.add_argument(
        "--reference_mode",
        default="official",
        choices=["official", "previous_outputs"],
        help="Reference source: examples/result official outputs or prior run outputs.",
    )
    ap.add_argument(
        "--previous_outputs_dir",
        default="output_checks/2026-02-08",
        help="Directory containing prior-run '<stem>_pred.md' files when using previous_outputs mode.",
    )
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument(
        "--cuda_no_fallback",
        action="store_true",
        help="Pass --cuda_no_fallback to run_onnx_static.py for CUDA-only execution.",
    )
    ap.add_argument(
        "--disable_kv_iobinding",
        action="store_true",
        help="Pass --disable_kv_iobinding to run_onnx_static.py.",
    )
    ap.add_argument(
        "--vision_policy",
        default="fp16",
        choices=["auto", "fp16", "quant", "table_quant"],
        help="Pass --vision_policy to run_onnx_static.py.",
    )
    ap.add_argument(
        "--disable_cuda_ep_tuning",
        action="store_true",
        help="Pass --disable_cuda_ep_tuning to run_onnx_static.py.",
    )
    ap.add_argument("--stems", default="seal,table,page")
    ap.add_argument("--out_dir", default="output_checks/quality_eval")
    ap.add_argument("--json_out", default="")
    ap.add_argument("--md_out", default="")
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir).resolve()
    examples_root = Path(args.examples_root).resolve()
    previous_outputs_dir = Path(args.previous_outputs_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = [s.strip() for s in args.stems.split(",") if s.strip()]
    if not stems:
        raise SystemExit("No stems provided.")

    rows = []
    for stem in stems:
        image_path = examples_root / "source" / f"{stem}.png"
        if args.reference_mode == "official":
            ref_path = examples_root / "result" / stem / "result.md"
        else:
            ref_path = previous_outputs_dir / f"{stem}_pred.md"
        if not image_path.exists():
            print(f"[skip] image missing: {image_path}")
            continue
        if not ref_path.exists():
            print(f"[skip] reference result missing: {ref_path}")
            continue

        pred_path = out_dir / f"{stem}_pred.md"
        task = infer_task(stem)
        print(f"[run] {stem} task={task}")
        elapsed_s = run_ocr(
            artifact_dir,
            image_path,
            pred_path,
            task,
            args.device,
            args.cuda_no_fallback,
            args.disable_kv_iobinding,
            args.vision_policy,
            args.disable_cuda_ep_tuning,
        )

        pred = pred_path.read_text(encoding="utf-8", errors="replace")
        ref = ref_path.read_text(encoding="utf-8", errors="replace")
        pred_n = normalize_text(pred)
        ref_n = normalize_text(ref)

        strict_cer = levenshtein(pred_n, ref_n) / max(1, len(ref_n))
        pred_r = relaxed_text(pred)
        ref_r = relaxed_text(ref)
        relaxed_cer = levenshtein(pred_r, ref_r) / max(1, len(ref_r))
        f1, prec, rec = char_f1_no_ws(pred, ref)

        rows.append((stem, strict_cer, relaxed_cer, f1, prec, rec, elapsed_s))
        print(
            f"[done] {stem} strict_cer={strict_cer:.4f} relaxed_cer={relaxed_cer:.4f} "
            f"char_f1={f1:.4f} elapsed_s={elapsed_s:.2f}"
        )

    if not rows:
        raise SystemExit("No comparable stems were processed.")

    mean_strict = sum(r[1] for r in rows) / len(rows)
    mean_relaxed = sum(r[2] for r in rows) / len(rows)
    mean_f1 = sum(r[3] for r in rows) / len(rows)
    mean_elapsed = sum(r[6] for r in rows) / len(rows)

    print("\n=== Summary ===")
    for stem, strict_cer, relaxed_cer, f1, prec, rec, elapsed_s in rows:
        print(
            f"{stem:10s} strict_cer={strict_cer:.4f} relaxed_cer={relaxed_cer:.4f} "
            f"char_f1={f1:.4f} precision={prec:.4f} recall={rec:.4f} elapsed_s={elapsed_s:.2f}"
        )
    print(
        f"mean       strict_cer={mean_strict:.4f} relaxed_cer={mean_relaxed:.4f} "
        f"char_f1={mean_f1:.4f} elapsed_s={mean_elapsed:.2f}"
    )

    if args.json_out:
        import json

        payload = {
            "artifact_dir": str(artifact_dir),
            "examples_root": str(examples_root),
            "reference_mode": args.reference_mode,
            "previous_outputs_dir": str(previous_outputs_dir),
            "device": args.device,
            "cuda_no_fallback": bool(args.cuda_no_fallback),
            "disable_kv_iobinding": bool(args.disable_kv_iobinding),
            "vision_policy": args.vision_policy,
            "disable_cuda_ep_tuning": bool(args.disable_cuda_ep_tuning),
            "stems": stems,
            "samples": [
                {
                    "stem": stem,
                    "strict_cer": strict_cer,
                    "relaxed_cer": relaxed_cer,
                    "char_f1": f1,
                    "precision": prec,
                    "recall": rec,
                    "elapsed_s": elapsed_s,
                }
                for stem, strict_cer, relaxed_cer, f1, prec, rec, elapsed_s in rows
            ],
            "mean": {
                "strict_cer": mean_strict,
                "relaxed_cer": mean_relaxed,
                "char_f1": mean_f1,
                "elapsed_s": mean_elapsed,
            },
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.md_out:
        lines = []
        lines.append("# Quality + Performance Report")
        lines.append("")
        lines.append(f"- artifact_dir: `{artifact_dir}`")
        lines.append(f"- examples_root: `{examples_root}`")
        lines.append(f"- reference_mode: `{args.reference_mode}`")
        if args.reference_mode == "previous_outputs":
            lines.append(f"- previous_outputs_dir: `{previous_outputs_dir}`")
        lines.append(f"- device: `{args.device}`")
        lines.append(f"- cuda_no_fallback: `{bool(args.cuda_no_fallback)}`")
        lines.append(f"- disable_kv_iobinding: `{bool(args.disable_kv_iobinding)}`")
        lines.append(f"- vision_policy: `{args.vision_policy}`")
        lines.append(f"- disable_cuda_ep_tuning: `{bool(args.disable_cuda_ep_tuning)}`")
        lines.append("")
        lines.append("| stem | strict_cer | relaxed_cer | char_f1 | precision | recall | elapsed_s |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for stem, strict_cer, relaxed_cer, f1, prec, rec, elapsed_s in rows:
            lines.append(
                f"| {stem} | {strict_cer:.4f} | {relaxed_cer:.4f} | {f1:.4f} | {prec:.4f} | {rec:.4f} | {elapsed_s:.2f} |"
            )
        lines.append(
            f"| mean | {mean_strict:.4f} | {mean_relaxed:.4f} | {mean_f1:.4f} | - | - | {mean_elapsed:.2f} |"
        )
        md_path = Path(args.md_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
