# WebGPU/WASM TypeScript Runtime

This folder now includes a TypeScript browser runtime that executes the split ONNX graphs with ORT Web.

## Files

- `manifest.web.json`: generated browser runtime manifest.
- `web_runtime_report.md`: size/memory report for browser profiles.
- `src/runtime.ts`: core TypeScript runtime with KV decode loop.
- `src/demo.ts`: static-browser demo entrypoint.
- `index.html`: minimal static page loading ORT Web + compiled TS output.

## Build

```bash
cd web_runtime
bun install
bun run build
```

## Run (static)

Serve the repo root with any static server, then open `web_runtime/index.html`.

Example:

```bash
python -m http.server 8080
```

Then open `http://localhost:8080/web_runtime/index.html`.

## Run Directly From Image (No Python Preprocessing at Runtime)

Generate fixed task presets once:

```bash
python scripts/export_web_task_presets.py \
  --manifest _local/variants/custom_w8_prefill8m_dual_vision/manifest.json \
  --out web_runtime/task_presets.json
```

Then in `web_runtime/index.html`:

- select an image file
- choose task/profile/EP
- run decode

The browser now performs image preprocessing (letterbox + normalize + GLM patchify) in TS and uses pre-tokenized task presets.

## Prepare Demo Input (Legacy JSON Path)

Generate a browser input JSON from an image:

```bash
python scripts/export_web_input_json.py \
  --manifest artifact_glm_ocr_web_split/manifest.json \
  --image examples/source/page.png \
  --task document \
  --out web_runtime/demo_input.json
```

Then choose `web_runtime/demo_input.json` in the demo page and run decode.

## Prepare KV-Only Artifact Bundle

For production KV decode deployments, create a trimmed artifact bundle that
removes the unused non-KV decode graph pair:

```bash
python scripts/prepare_kv_only_artifact.py \
  --artifact_dir artifact_glm_ocr_web_split \
  --out_dir _local/variants/kv_only_prepared
```

Use `--hardlink` to avoid duplicating large files on the same filesystem.

## Input JSON for Demo

The demo expects one JSON file with this shape:

```json
{
  "input_ids": [59280, 123, 456],
  "attention_mask": [1, 1, 1],
  "image_grid_thw": [1, 30, 30],
  "pixel_values": [0.0, 0.1, 0.2],
  "seq_len": 3,
  "t_img": 900,
  "hidden_size": 1536,
  "max_new_tokens": 256
}
```

Notes:

- `pixel_values` is required for `full_browser_kv` and must be patchified `[tokens, features]` format.
- `image_embeds` can be provided instead of `pixel_values` for hybrid mode.

## Profiles

Defined in `manifest.web.json`:

- `full_browser_kv`: 100% browser execution (very high memory footprint).
- `hybrid_server_vision_client_kv`: server vision, browser text decode.
- `hybrid_server_vision_client_prefill_only`: simpler hybrid profile.

## Performance + Quality Notes

- Runtime uses a fast greedy path for deterministic decode to reduce CPU overhead without quality changes.
- Reuses feed buffers and KV tensor mappings in decode loops.
- Optional per-token embed cache avoids repeated static-shape embed ONNX calls for repeated token IDs.
- Optional task-aware vision policy supports dual-vision manifests (`graphs.vision` + `graphs.vision_quant`):
  - Default runtime behavior is `fp16` (quality-safe baseline).
  - `visionPolicy: "auto"`: `text` uses `vision`, others use `vision_quant` when present.
  - `visionPolicy: "fp16"`: always use `vision`.
  - `visionPolicy: "quant"`: always use `vision_quant` when present.
  - `visionPolicy: "table_quant"`: use `vision_quant` only for `table` task.
  - Recommended production setting from current tests: `table_quant`.
- For production, keep sessions lazy-loaded and only retain required graphs.
- Browser memory remains the hard limit; `full_browser_kv` is mainly for high-memory environments.
