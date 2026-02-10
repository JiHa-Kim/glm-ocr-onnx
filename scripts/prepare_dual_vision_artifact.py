#!/usr/bin/env python3
"""Build artifact with both fp16 and quantized vision graphs.

Inputs:
- base_artifact_dir: working artifact that already has desired decode graphs.
- quant_vision_artifact_dir: artifact containing quantized vision graph.

Output:
- out_dir with:
  - graphs.vision (from base artifact, unchanged)
  - graphs.vision_quant (copied from quant vision artifact as quant/glm_ocr_vision_quant.onnx)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import onnx

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def choose_manifest_path(artifact_dir: Path) -> Path:
    p = artifact_dir / "manifest.json"
    if p.exists():
        return p
    p2 = artifact_dir.parent / "manifest.json"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"manifest.json not found under {artifact_dir}")


def resolve_graph_path(artifact_dir: Path, graph_ref: str) -> Path:
    p = Path(graph_ref)
    if p.is_absolute() and p.exists():
        return p
    c1 = artifact_dir / p
    if c1.exists():
        return c1
    c2 = artifact_dir / "fp16" / p.name
    if c2.exists():
        return c2
    c3 = artifact_dir / "fp32" / p.name
    if c3.exists():
        return c3
    return c1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_artifact_dir", default="_local/variants/custom_w8_prefill_8m")
    ap.add_argument(
        "--quant_vision_artifact_dir",
        default="_local/variants/custom_w8_prefill8m_vision2m",
    )
    ap.add_argument("--out_dir", default="_local/variants/custom_w8_prefill8m_dual_vision")
    args = ap.parse_args()

    base = Path(args.base_artifact_dir).resolve()
    qv = Path(args.quant_vision_artifact_dir).resolve()
    out = Path(args.out_dir).resolve()

    base_manifest = choose_manifest_path(base)
    qv_manifest = choose_manifest_path(qv)
    cfg_base = read_json(base_manifest)
    cfg_qv = read_json(qv_manifest)

    if "vision" not in cfg_base.get("graphs", {}):
        raise RuntimeError("base artifact missing graphs.vision")
    if "vision" not in cfg_qv.get("graphs", {}):
        raise RuntimeError("quant vision artifact missing graphs.vision")

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out)

    qv_vision_path = resolve_graph_path(qv, str(cfg_qv["graphs"]["vision"]))
    if not qv_vision_path.exists():
        raise FileNotFoundError(f"quant vision graph missing: {qv_vision_path}")
    qv_data_path = qv_vision_path.with_suffix(qv_vision_path.suffix + ".data")
    if not qv_data_path.exists():
        raise FileNotFoundError(f"quant vision sidecar missing: {qv_data_path}")

    out_quant = out / "quant"
    out_quant.mkdir(parents=True, exist_ok=True)
    out_vision_q = out_quant / "glm_ocr_vision_quant.onnx"
    out_vision_q_data = out_quant / "glm_ocr_vision_quant.onnx.data"
    # Re-save external data with a local sidecar name that matches the new file.
    m_qv = onnx.load(str(qv_vision_path), load_external_data=True)
    if out_vision_q_data.exists():
        out_vision_q_data.unlink()
    onnx.save_model(
        m_qv,
        str(out_vision_q),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=out_vision_q.name + ".data",
        size_threshold=1024,
    )

    out_cfg = read_json(out / "manifest.json")
    out_cfg.setdefault("graphs", {})
    out_cfg["graphs"]["vision_quant"] = "quant/glm_ocr_vision_quant.onnx"
    out_cfg.setdefault("notes", [])
    out_cfg["notes"].append(
        "dual-vision artifact: graphs.vision=fp16 and graphs.vision_quant=quantized."
    )
    write_json(out / "manifest.json", out_cfg)

    print(out)
    print(out / "manifest.json")


if __name__ == "__main__":
    main()
