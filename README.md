# GLM OCR ONNX Runtime

## Attribution

This project is built on top of the original GLM-OCR release from Z.ai:

- Source repository: https://github.com/zai-org/GLM-OCR
- Model page: https://huggingface.co/zai-org/GLM-OCR

Static OCR inference utilities for split GLM-4.6V/GLM-OCR ONNX artifacts.

This repository focuses on running pre-exported ONNX graphs (not exporting from PyTorch) for:
- Python inference with ONNX Runtime (`run_onnx_static.py`)
- Browser inference with ORT Web (`web_runtime/`)
- Artifact packaging, quantization, and web-manifest prep (`scripts/`)

## What this repo does

- Runs OCR from a single image or layout regions using split ONNX graphs.
- Supports KV-cache decode path (`decode_prefill_kv` + `decode_step_kv`) for better decode throughput.
- Supports optional dual-vision artifacts (`graphs.vision` + `graphs.vision_quant`) with task-aware policy.
- Generates web runtime manifests and browser demo inputs/presets.

## Repository layout

- `run_onnx_static.py`: main Python entrypoint for static ONNX OCR inference.
- `scripts/`: artifact utilities (quantization, packaging, web prep, quality comparison).
- `web_runtime/`: TypeScript runtime + static demo page for ORT Web.
- `examples/source/`: sample input images/PDF.
- `examples/result/`: reference output examples.

## Prerequisites

- Python 3.14+ (per `pyproject.toml`)
- A split GLM OCR ONNX artifact directory containing `manifest.json` and graph files
- For GPU runs: CUDA-compatible ONNX Runtime setup

Optional:
- `uv` for environment management
- `bun` (for building `web_runtime`)

## Install

Using `uv`:

```bash
uv sync
```

Using pip:

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e .
```

## Artifact expectations

`run_onnx_static.py` expects an artifact directory via `--artifact_dir` and resolves `manifest.json` from either:
- `<artifact_dir>/manifest.json`, or
- `<artifact_dir>/../manifest.json`

Typical required manifest graph keys:
- `vision`
- `embed`
- `rope` (or `prompt_profiles[*].rope`)
- KV path: `decode_prefill_kv` + `decode_step_kv`
- Non-KV fallback path: `decode`

Optional:
- `vision_quant` (for `--vision_policy`)

## Quick start (Python)

Single image OCR:

```bash
python run_onnx_static.py \
  --artifact_dir artifact_glm_ocr_web_split \
  --image examples/source/page.png \
  --task document \
  --device cpu \
  --out_text output.md
```

CUDA OCR:

```bash
python run_onnx_static.py \
  --artifact_dir artifact_glm_ocr_web_split \
  --image examples/source/table.png \
  --task table \
  --device cuda \
  --official_quality
```

Layout-driven OCR (region-wise from detector JSON):

```bash
python run_onnx_static.py \
  --artifact_dir artifact_glm_ocr_web_split \
  --image examples/source/page.png \
  --layout_json path/to/layout.json \
  --layout_page_index 0 \
  --layout_out_json layout_ocr.json \
  --device cpu
```

## Main CLI options (`run_onnx_static.py`)

Core:
- `--artifact_dir` (required)
- `--image` (required)
- `--task {document,text,table,formula}`
- `--out_text <path>`

Generation control:
- `--max_new_tokens`
- `--max_new_tokens_{text,table,formula,document}`
- `--adaptive_budgets 128,256,...`
- `--official_quality` (forces deterministic, quality-oriented settings)
- `--do_sample --temperature --top_p --top_k --num_candidates --seed`
- `--repetition_penalty --no_repeat_ngram_size`

Prompt/rope profile:
- `--prompt`
- `--rope_profile`
- `--allow_prompt_override`

Layout mode:
- `--layout_json`
- `--layout_page_index`
- `--region_pad`
- `--layout_out_json`

Runtime/device:
- `--device {cpu,cuda}`
- `--cuda_no_fallback`
- `--disable_cuda_ep_tuning`
- `--disable_kv_cache`
- `--disable_kv_iobinding`
- `--fast_embed_lookup`
- `--vision_policy {auto,fp16,quant,table_quant}`

## Quality evaluation helper

Compare current outputs against `examples/result/*/result.md`:

```bash
python scripts/compare_official_quality.py \
  --artifact_dir artifact_glm_ocr_web_split \
  --examples_root examples \
  --device cuda \
  --stems seal,table,page \
  --out_dir output_checks/quality_eval
```

## Web runtime

See `web_runtime/README.md` for detailed browser instructions.

Basic build:

```bash
cd web_runtime
bun install
bun run build
```

Serve repository root, then open `web_runtime/index.html`.

Prepare web manifest from an artifact:

```bash
python scripts/prepare_web_runtime.py \
  --artifact_dir artifact_glm_ocr_web_split \
  --out_dir web_runtime \
  --base_url ./artifact_glm_ocr_web_split/ \
  --rope_profile document
```

Prepare fixed task presets for browser-side preprocessing:

```bash
python scripts/export_web_task_presets.py \
  --manifest artifact_glm_ocr_web_split/manifest.json \
  --out web_runtime/task_presets.json
```

## Artifact utility scripts

- `scripts/prepare_kv_only_artifact.py`: strip non-KV decode graph and keep KV-compatible bundle.
- `scripts/quantize_custom_w8.py`: custom weight-only INT8 quantization for selected graphs.
- `scripts/prepare_dual_vision_artifact.py`: merge base artifact with quantized vision graph (`vision_quant`).
- `scripts/sync_hf_upload_bundle.py`: build a self-contained `hf_upload_bundle` for static hosting/Hugging Face upload.
- `scripts/export_web_input_json.py`: create legacy web demo JSON input from image.

## Troubleshooting

- `manifest.json not found`: verify `--artifact_dir` points to the artifact root (or its child containing `fp16/`, `quant/`, etc.).
- Prompt/profile mismatch errors: use `--rope_profile` matching your exported `prompt_profiles`.
- CUDA provider fallback/partitioning surprises: try `--cuda_no_fallback` and inspect your ORT CUDA install.
- Large browser memory usage: prefer hybrid profiles in `manifest.web.json`; avoid eagerly loading all graphs.

## License

MIT (see `LICENSE`).
