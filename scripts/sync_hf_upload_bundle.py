#!/usr/bin/env python3
"""Sync hf_upload_bundle from a source artifact and regenerate web manifest.

This keeps the upload bundle self-contained and consistent with the selected
optimized artifact (including optional vision_quant graph).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List

README_TEXT = """# GLM-OCR HF Upload Bundle

This folder is a self-contained static bundle for browser inference with ORT Web (WebGPU/WASM).

## Contents

- `manifest.json`: runtime artifact manifest for Python/static flows.
- `manifest.web.json`: ORT Web wiring manifest (self-contained `base_url: "./"`).
- `fp16/`: fp16 ONNX graphs + external data.
- `quant/`: quantized ONNX graphs + external data.
- `index.html` + `dist/*.js`: browser demo UI/runtime.

## Run Locally

From repo root:

```bash
python -m http.server 8080
```

Open:

`http://localhost:8080/hf_upload_bundle/index.html`

## Demo Input

Use `scripts/export_web_input_json.py` to generate a JSON input from an image.

Example:

```bash
python scripts/export_web_input_json.py \\
  --manifest hf_upload_bundle/manifest.json \\
  --image examples/source/page.png \\
  --task document \\
  --out hf_upload_bundle/demo_input.json
```

Then load `hf_upload_bundle/demo_input.json` in the web demo.

## Hugging Face Upload

Upload the full `hf_upload_bundle/` directory as the repository root of your HF space/repo.

Because `manifest.web.json` uses `base_url: "./"`, assets resolve correctly from static hosting.
"""

GITIGNORE_TEXT = """web_runtime_report.md
fp16/*
quant/*
!quant/glm_ocr_vision_quant.onnx
!quant/glm_ocr_vision_quant.onnx.data
"""


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


def iter_graph_files(artifact_dir: Path, graph_ref: str) -> Iterable[Path]:
    p = resolve_graph_path(artifact_dir, graph_ref)
    yield p
    d = p.with_suffix(p.suffix + ".data")
    if d.exists():
        yield d


def graph_refs_to_copy(cfg: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    graphs = cfg.get("graphs", {})
    for k in ["vision", "vision_quant", "embed", "decode_prefill_kv", "decode_step_kv"]:
        v = str(graphs.get(k, "")).strip()
        if v:
            refs.append(v)
    pp = cfg.get("prompt_profiles", {})
    for info in pp.values():
        rope = str(info.get("rope", "")).strip()
        if rope:
            refs.append(rope)
    seen = set()
    out: List[str] = []
    for r in refs:
        if r and r not in seen:
            out.append(r)
            seen.add(r)
    return out


def copy_or_link(src: Path, dst: Path, hardlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def copy_web_demo_assets(repo_root: Path, bundle_dir: Path) -> None:
    web_runtime = repo_root / "web_runtime"
    index_src = web_runtime / "index.html"
    dist_src = web_runtime / "dist"
    if index_src.exists():
        shutil.copy2(index_src, bundle_dir / "index.html")
    if dist_src.exists():
        shutil.copytree(dist_src, bundle_dir / "dist", dirs_exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_artifact_dir", default="_local/variants/custom_w8_prefill8m_dual_vision")
    ap.add_argument("--bundle_dir", default="hf_upload_bundle")
    ap.add_argument("--hardlink", action="store_true")
    ap.add_argument("--rope_profile", default="document")
    args = ap.parse_args()

    src = Path(args.source_artifact_dir).resolve()
    bundle = Path(args.bundle_dir).resolve()
    src_manifest = choose_manifest_path(src)
    cfg = read_json(src_manifest)

    refs = graph_refs_to_copy(cfg)

    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True, exist_ok=True)

    copied = set()
    for ref in refs:
        for f in iter_graph_files(src, ref):
            if not f.exists():
                raise FileNotFoundError(f"missing source file: {f}")
            rel = f.relative_to(src)
            dst = bundle / rel
            if dst in copied:
                continue
            copy_or_link(f, dst, hardlink=args.hardlink)
            copied.add(dst)

    # Bundle manifest.json mirrors source and uses bundle-local graph refs.
    write_json(bundle / "manifest.json", cfg)

    # Generate self-contained web manifest with base_url "./".
    subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            "scripts/prepare_web_runtime.py",
            "--artifact_dir",
            str(bundle),
            "--out_dir",
            str(bundle),
            "--base_url",
            "./",
            "--rope_profile",
            args.rope_profile,
        ],
        check=True,
    )

    copy_web_demo_assets(Path(__file__).resolve().parent.parent, bundle)
    (bundle / "README.md").write_text(README_TEXT, encoding="utf-8")
    (bundle / ".gitignore").write_text(GITIGNORE_TEXT, encoding="utf-8")

    print(bundle)
    print(bundle / "manifest.json")
    print(bundle / "manifest.web.json")


if __name__ == "__main__":
    main()
