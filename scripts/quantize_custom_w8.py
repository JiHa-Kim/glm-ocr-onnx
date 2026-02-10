#!/usr/bin/env python3
"""Custom weight-only INT8 quantization for large MatMul/Gemm weights.

This avoids DynamicQuantizeLinear-on-activation issues by quantizing only
constant weight initializers used by MatMul/Gemm and dequantizing them back
to FP16 via:
  int8_weight --DequantizeLinear--> float32 --Cast--> float16
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def choose_manifest_path(artifact_dir: Path) -> Path:
    p = artifact_dir / "manifest.json"
    if p.exists():
        return p
    p2 = artifact_dir.parent / "manifest.json"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"manifest.json not found under {artifact_dir}")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_or_link(src: Path, dst: Path, hardlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def resolve_graph_path(artifact_dir: Path, graph_ref: str) -> Path:
    p = Path(graph_ref)
    if p.is_absolute():
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


def iter_graph_files(artifact_dir: Path, graph_ref: str) -> Iterable[Path]:
    p = resolve_graph_path(artifact_dir, graph_ref)
    yield p
    d = p.with_suffix(p.suffix + ".data")
    if d.exists():
        yield d


def graph_refs_to_keep(cfg: Dict[str, Any]) -> List[str]:
    graphs = cfg.get("graphs", {})
    refs = []
    for k in ["vision", "embed", "decode_prefill_kv", "decode_step_kv"]:
        if k in graphs:
            refs.append(str(graphs[k]))
    pp = cfg.get("prompt_profiles", {})
    for _, info in pp.items():
        rope = str(info.get("rope", "")).strip()
        if rope:
            refs.append(rope)
    rope_default = str(graphs.get("rope", "")).strip()
    if rope_default:
        refs.append(rope_default)
    out = []
    seen = set()
    for r in refs:
        if r and r not in seen:
            out.append(r)
            seen.add(r)
    return out


def quantize_model_weights_w8(src: Path, dst: Path, min_numel: int) -> Dict[str, Any]:
    m = onnx.load(str(src), load_external_data=True)
    init_map = {i.name: i for i in m.graph.initializer}

    candidates = set()
    for n in m.graph.node:
        if n.op_type in {"MatMul", "Gemm"} and len(n.input) >= 2 and n.input[1] in init_map:
            candidates.add(n.input[1])

    kept = []
    new_inits = []
    replaced: Dict[str, str] = {}

    skipped_small = 0
    total_quantized_numel = 0
    for name, init in init_map.items():
        if name not in candidates:
            kept.append(init)
            continue
        arr = numpy_helper.to_array(init)
        if arr.dtype != np.float16:
            kept.append(init)
            continue
        if int(arr.size) < int(min_numel):
            kept.append(init)
            skipped_small += 1
            continue
        arr32 = arr.astype(np.float32)
        amax = float(np.max(np.abs(arr32)))
        scale = 1.0 if (not np.isfinite(amax) or amax == 0.0) else (amax / 127.0)
        q = np.clip(np.round(arr32 / scale), -127, 127).astype(np.int8)

        qn = name + "__qint8"
        sn = name + "__scale"
        zn = name + "__zp"
        dqn = name + "__deq_f32"
        cn = name + "__deq_f16"

        new_inits.append(numpy_helper.from_array(q, qn))
        new_inits.append(numpy_helper.from_array(np.array(scale, dtype=np.float32), sn))
        new_inits.append(numpy_helper.from_array(np.array(0, dtype=np.int8), zn))
        replaced[name] = cn
        total_quantized_numel += int(arr.size)

    inserted = set()
    new_nodes = []
    for n in m.graph.node:
        node = onnx.NodeProto()
        node.CopyFrom(n)
        for idx, inp in enumerate(list(node.input)):
            if inp not in replaced:
                continue
            if not ((node.op_type == "MatMul" and idx == 1) or (node.op_type == "Gemm" and idx == 1)):
                continue
            if inp not in inserted:
                qn = inp + "__qint8"
                sn = inp + "__scale"
                zn = inp + "__zp"
                dqn = inp + "__deq_f32"
                cn = inp + "__deq_f16"
                new_nodes.append(helper.make_node("DequantizeLinear", [qn, sn, zn], [dqn], name=inp + "__DQL"))
                new_nodes.append(
                    helper.make_node(
                        "Cast",
                        [dqn],
                        [cn],
                        name=inp + "__CastF16",
                        to=TensorProto.FLOAT16,
                    )
                )
                inserted.add(inp)
            node.input[idx] = replaced[inp]
        new_nodes.append(node)

    del m.graph.initializer[:]
    m.graph.initializer.extend(kept + new_inits)
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)

    out_data = dst.with_suffix(dst.suffix + ".data")
    if out_data.exists():
        out_data.unlink()
    onnx.save_model(
        m,
        str(dst),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=dst.name + ".data",
        size_threshold=1024,
    )

    return {
        "replaced_weights": len(replaced),
        "skipped_small_weights": int(skipped_small),
        "quantized_numel": int(total_quantized_numel),
        "min_numel": int(min_numel),
        "out_data_bytes": out_data.stat().st_size if out_data.exists() else 0,
    }


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", default="artifact_glm_ocr_web_split")
    ap.add_argument("--out_dir", default="_local/variants/custom_w8_all")
    ap.add_argument("--hardlink", action="store_true")
    ap.add_argument(
        "--targets",
        default="vision,decode_prefill_kv,decode_step_kv,embed",
        help="Comma-separated graph keys to quantize (from manifest graphs).",
    )
    ap.add_argument(
        "--min_numel",
        type=int,
        default=4096,
        help="Only quantize FP16 MatMul/Gemm weights with at least this many elements.",
    )
    args = ap.parse_args()

    src = Path(args.artifact_dir).resolve()
    out = Path(args.out_dir).resolve()
    manifest_path = choose_manifest_path(src)
    cfg = read_json(manifest_path)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    target_set = set(targets)

    if out.exists():
        shutil.rmtree(out)
    (out / "fp16").mkdir(parents=True, exist_ok=True)

    refs = graph_refs_to_keep(cfg)
    quantize_names = set()
    for k in target_set:
        ref = str(cfg.get("graphs", {}).get(k, "")).strip()
        if not ref:
            continue
        n = Path(ref).name
        quantize_names.add(n)
        quantize_names.add(n + ".data")

    copied = set()
    for ref in refs:
        for f in iter_graph_files(src, ref):
            if not f.exists():
                raise FileNotFoundError(f"missing graph file: {f}")
            rel = f.relative_to(src)
            dst = out / rel
            if dst in copied:
                continue
            # Target files must be copied (not hardlinked) because we edit them.
            use_hardlink = args.hardlink and (f.name not in quantize_names)
            copy_or_link(f, dst, hardlink=use_hardlink)
            copied.add(dst)

    helper_img = src / "static_export_image.png"
    if helper_img.exists():
        shutil.copy2(helper_img, out / helper_img.name)

    out_cfg = json.loads(json.dumps(cfg))
    out_cfg["graphs"].pop("decode", None)
    out_cfg.setdefault("notes", [])

    report: Dict[str, Any] = {
        "source_dir": str(src),
        "out_dir": str(out),
        "targets": targets,
        "results": {},
    }

    for k in targets:
        ref = out_cfg.get("graphs", {}).get(k)
        if not ref:
            continue
        p = resolve_graph_path(out, str(ref))
        r = quantize_model_weights_w8(p, p, min_numel=max(1, int(args.min_numel)))
        report["results"][k] = r
        out_cfg["notes"].append(f"custom_w8: quantized MatMul/Gemm weights for graph '{k}'.")

    write_json(out / "manifest.json", out_cfg)
    report["source_bytes"] = dir_size(src)
    report["out_bytes"] = dir_size(out)
    report["saved_bytes"] = report["source_bytes"] - report["out_bytes"]
    write_json(out / "quantize_custom_w8_report.json", report)
    print(out)
    print(out / "quantize_custom_w8_report.json")
    print(f"saved_bytes={report['saved_bytes']}")


if __name__ == "__main__":
    main()
