#!/usr/bin/env python3
"""Prepare a WebGPU/WASM runtime manifest from split ONNX artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_manifest_path(artifact_dir: Path) -> Path:
    p = artifact_dir / "manifest.json"
    if p.exists():
        return p
    p2 = artifact_dir.parent / "manifest.json"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"manifest.json not found under {artifact_dir}")


def bytes_for(cfg: Dict[str, Any], key: str) -> int:
    ent = cfg.get(key, {})
    return int(ent.get("bytes", 0)) + int(ent.get("data_bytes", 0))


def graph_ref(cfg: Dict[str, Any], key: str) -> str:
    return str(cfg["graphs"][key])


def rope_ref(cfg: Dict[str, Any], profile: str) -> str:
    pp = cfg.get("prompt_profiles", {})
    if profile not in pp:
        raise KeyError(
            f"unknown profile '{profile}', available: {', '.join(sorted(pp.keys()))}"
        )
    return str(pp[profile]["rope"])


def fmt_gb(n: int) -> float:
    return float(n) / (1024.0**3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", default="artifact_glm_ocr_web_split")
    ap.add_argument("--out_dir", default="web_runtime")
    ap.add_argument("--base_url", default="./artifact_glm_ocr_web_split/")
    ap.add_argument("--rope_profile", default="document")
    ap.add_argument("--max_graph_bytes_warn_gb", type=float, default=2.0)
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = choose_manifest_path(artifact_dir)
    cfg = read_json(manifest_path)

    required = [
        "vision",
        "embed",
        "decode_prefill_kv",
        "decode_step_kv",
    ]
    for k in required:
        if k not in cfg.get("graphs", {}):
            raise KeyError(f"manifest graphs missing required key: {k}")

    rope = rope_ref(cfg, args.rope_profile)
    graphs = {
        "vision": graph_ref(cfg, "vision"),
        "embed": graph_ref(cfg, "embed"),
        "rope": rope,
        "decode_prefill_kv": graph_ref(cfg, "decode_prefill_kv"),
        "decode_step_kv": graph_ref(cfg, "decode_step_kv"),
    }
    if "vision_quant" in cfg.get("graphs", {}):
        graphs["vision_quant"] = graph_ref(cfg, "vision_quant")

    # Weight blob budgeting for common load strategies.
    budget_bytes = {
        "full_browser_kv": (
            bytes_for(cfg, "vision")
            + bytes_for(cfg, "embed")
            + bytes_for(cfg, "rope_" + args.rope_profile)
            + bytes_for(cfg, "decode_prefill_kv")
            + bytes_for(cfg, "decode_step_kv")
        ),
        "full_browser_prefill_only": (
            bytes_for(cfg, "vision")
            + bytes_for(cfg, "embed")
            + bytes_for(cfg, "rope_" + args.rope_profile)
            + bytes_for(cfg, "decode_prefill_kv")
        ),
        "hybrid_server_vision_client_kv": (
            bytes_for(cfg, "embed")
            + bytes_for(cfg, "rope_" + args.rope_profile)
            + bytes_for(cfg, "decode_prefill_kv")
            + bytes_for(cfg, "decode_step_kv")
        ),
        "hybrid_server_vision_client_prefill_only": (
            bytes_for(cfg, "embed")
            + bytes_for(cfg, "rope_" + args.rope_profile)
            + bytes_for(cfg, "decode_prefill_kv")
        ),
    }

    per_graph_bytes = {
        "vision": bytes_for(cfg, "vision"),
        "embed": bytes_for(cfg, "embed"),
        "rope_profile": bytes_for(cfg, "rope_" + args.rope_profile),
        "decode_prefill_kv": bytes_for(cfg, "decode_prefill_kv"),
        "decode_step_kv": bytes_for(cfg, "decode_step_kv"),
    }
    if "vision_quant" in cfg.get("graphs", {}):
        per_graph_bytes["vision_quant"] = bytes_for(cfg, "vision_quant")
    warnings: List[str] = []
    warn_bytes = int(args.max_graph_bytes_warn_gb * (1024.0**3))
    for name, b in per_graph_bytes.items():
        if b > warn_bytes:
            warnings.append(
                f"{name} is {fmt_gb(b):.2f} GB; likely too large for direct browser startup on most clients."
            )

    web_manifest = {
        "model_id": cfg["model_id"],
        "dtype": cfg["dtype"],
        "opset": int(cfg["opset"]),
        "max_seq_len": int(cfg["max_seq_len"]),
        "image_size": cfg["image_size"],
        "image_token_id": int(cfg["image_token_id"]),
        "eos_token_ids": cfg.get("eos_token_ids", []),
        "kv_cache": cfg.get("kv_cache", {}),
        "prompt_profile": args.rope_profile,
        "base_url": args.base_url,
        "graphs": graphs,
        "runtime": {
            "webgpu": {
                "executionProviders": ["webgpu"],
                "logSeverityLevel": 3,
            },
            "wasm": {
                "executionProviders": ["wasm"],
                "logSeverityLevel": 3,
            },
        },
        "profiles": {
            "full_browser_kv": {
                "description": "All sessions in browser, KV prefill+step decoding.",
                "graphs": [
                    "vision",
                    "embed",
                    "rope",
                    "decode_prefill_kv",
                    "decode_step_kv",
                ],
                "requires_image_embeds_input": False,
                "estimated_weight_gb": round(fmt_gb(budget_bytes["full_browser_kv"]), 3),
            },
            "hybrid_server_vision_client_kv": {
                "description": "Vision/image embedding runs on server; browser runs text decode with KV.",
                "graphs": [
                    "embed",
                    "rope",
                    "decode_prefill_kv",
                    "decode_step_kv",
                ],
                "requires_image_embeds_input": True,
                "estimated_weight_gb": round(
                    fmt_gb(budget_bytes["hybrid_server_vision_client_kv"]), 3
                ),
            },
            "hybrid_server_vision_client_prefill_only": {
                "description": "Vision on server; browser uses decode_prefill_kv only (simpler, slower token loop).",
                "graphs": [
                    "embed",
                    "rope",
                    "decode_prefill_kv",
                ],
                "requires_image_embeds_input": True,
                "estimated_weight_gb": round(
                    fmt_gb(budget_bytes["hybrid_server_vision_client_prefill_only"]), 3
                ),
            },
        },
        "load_order": [
            "vision",
            *(
                ["vision_quant"]
                if "vision_quant" in cfg.get("graphs", {})
                else []
            ),
            "embed",
            "rope",
            "decode_prefill_kv",
            "decode_step_kv",
        ],
        "memory_budget_gb": {
            k: round(fmt_gb(v), 3) for k, v in budget_bytes.items()
        },
        "per_graph_size_gb": {
            k: round(fmt_gb(v), 3) for k, v in per_graph_bytes.items()
        },
        "warnings": warnings,
        "notes": [
            "This manifest is for ORT Web (WebGPU/WASM) session wiring.",
            "Prefer lazy session creation; do not load unused graphs.",
            "Use rope graph matching the exported prompt profile.",
            "For browser deployments, hybrid mode is usually required at current model size.",
        ],
    }

    out_json = out_dir / "manifest.web.json"
    out_json.write_text(json.dumps(web_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Web Runtime Port Report")
    lines.append("")
    lines.append(f"- source_manifest: `{manifest_path}`")
    lines.append(f"- output_manifest: `{out_json}`")
    lines.append(f"- prompt_profile: `{args.rope_profile}`")
    lines.append("")
    lines.append("## Per-Graph Size (GB)")
    for k, v in web_manifest["per_graph_size_gb"].items():
        lines.append(f"- {k}: {v:.3f} GB")
    lines.append("")
    lines.append("## Typical Session Load Budgets (GB)")
    for k, v in web_manifest["memory_budget_gb"].items():
        lines.append(f"- {k}: {v:.3f} GB")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        for w in warnings:
            lines.append(f"- {w}")

    out_md = out_dir / "web_runtime_report.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_json)
    print(out_md)


if __name__ == "__main__":
    main()
