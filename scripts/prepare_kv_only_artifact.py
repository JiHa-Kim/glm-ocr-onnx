#!/usr/bin/env python3
"""Create a KV-only artifact bundle from split GLM-OCR ONNX exports.

This keeps only the graphs needed by the KV decode path and removes the
non-KV decode graph pair to reduce package size without changing model output.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Set


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
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


def graph_refs_to_keep(cfg: Dict[str, Any]) -> Set[str]:
    graphs = cfg.get("graphs", {})
    required = {"vision", "embed", "decode_prefill_kv", "decode_step_kv"}
    missing = sorted(k for k in required if k not in graphs)
    if missing:
        raise KeyError(f"manifest graphs missing required key(s): {', '.join(missing)}")

    refs: Set[str] = set()
    for k in required:
        refs.add(str(graphs[k]))

    # Keep all prompt-profile rope variants to preserve task/profile behavior.
    prompt_profiles = cfg.get("prompt_profiles", {})
    if not prompt_profiles:
        refs.add(str(graphs.get("rope", "")))
    else:
        for _, info in prompt_profiles.items():
            rope = str(info.get("rope", "")).strip()
            if rope:
                refs.add(rope)

    # Fallback rope in older manifests.
    rope_default = str(graphs.get("rope", "")).strip()
    if rope_default:
        refs.add(rope_default)

    refs.discard("")
    return refs


def iter_graph_files(artifact_dir: Path, graph_ref: str) -> Iterable[Path]:
    model_path = resolve_graph_path(artifact_dir, graph_ref)
    yield model_path
    data_path = model_path.with_suffix(model_path.suffix + ".data")
    if data_path.exists():
        yield data_path


def copy_or_link(src: Path, dst: Path, hardlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def path_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", default="artifact_glm_ocr_web_split")
    ap.add_argument("--out_dir", default="_local/variants/kv_only_prepared")
    ap.add_argument(
        "--hardlink",
        action="store_true",
        help="Use hardlinks for large files instead of copying.",
    )
    args = ap.parse_args()

    src_dir = Path(args.artifact_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = choose_manifest_path(src_dir)
    cfg = read_json(manifest_path)
    refs = graph_refs_to_keep(cfg)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "fp16").mkdir(parents=True, exist_ok=True)

    # Copy optional helper image if present.
    helper_img = src_dir / "static_export_image.png"
    if helper_img.exists():
        shutil.copy2(helper_img, out_dir / helper_img.name)

    copied: Set[Path] = set()
    for graph_ref in sorted(refs):
        for src_file in iter_graph_files(src_dir, graph_ref):
            if not src_file.exists():
                raise FileNotFoundError(f"referenced graph file missing: {src_file}")
            rel = src_file.relative_to(src_dir)
            dst_file = out_dir / rel
            if dst_file in copied:
                continue
            copy_or_link(src_file, dst_file, hardlink=args.hardlink)
            copied.add(dst_file)

    # Rewrite manifest for KV-only packaging.
    out_cfg = json.loads(json.dumps(cfg))
    out_cfg_graphs = out_cfg.setdefault("graphs", {})
    out_cfg_graphs.pop("decode", None)
    out_cfg.setdefault("notes", [])
    out_cfg["notes"].append(
        "KV-only artifact: decode graph removed; use decode_prefill_kv + decode_step_kv path."
    )
    write_json(out_dir / "manifest.json", out_cfg)

    in_bytes = path_size_bytes(src_dir)
    out_bytes = path_size_bytes(out_dir)
    saved = in_bytes - out_bytes
    print(f"source: {src_dir}")
    print(f"output: {out_dir}")
    print(f"source_bytes: {in_bytes}")
    print(f"output_bytes: {out_bytes}")
    print(f"saved_bytes: {saved}")


if __name__ == "__main__":
    main()

