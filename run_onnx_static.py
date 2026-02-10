#!/usr/bin/env python3
"""
Static GLM-OCR ONNX runtime (no PyTorch model load).

This script is designed for fixed-shape split-ONNX artifacts intended for
WASM/WebGPU-style deployment flows:
  1) vision: pixel_values + image_grid_thw -> image_embeds
  2) embed : input_ids -> token_embeds
  3) rope  : input_ids + image_grid_thw + attention_mask -> position_ids
  4) decode: inputs_embeds + attention_mask + position_ids -> logits
"""

import argparse
import json
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import (
    AutoTokenizer,
    Glm46VImageProcessor,
    Glm46VProcessor,
    Glm46VVideoProcessor,
)

try:
    ort.set_default_logger_severity(3)  # ty: ignore
except Exception:
    pass

if hasattr(sys.stdout, "reconfigure"):
    # Avoid cp1252 crashes on Windows when decoded text contains non-ASCII.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROMPT_PRESETS = {
    "document": (
        "Recognize the text in the image and output in Markdown format. "
        "Preserve the original layout (headings/paragraphs/tables/formulas). "
        "Do not fabricate content that does not exist in the image."
    ),
    "text": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}

LABEL_TASK_MAPPING = {
    "table": {
        "table",
    },
    "formula": {
        "display_formula",
        "inline_formula",
        "formula",
    },
    "text": {
        "abstract",
        "algorithm",
        "content",
        "doc_title",
        "figure_title",
        "paragraph_title",
        "reference_content",
        "text",
        "vertical_text",
        "vision_footnote",
        "seal",
        "formula_number",
    },
}


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_graph_path(artifact_dir: Path, manifest_path: Path, graph_ref: str) -> Path:
    p = Path(graph_ref)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        artifact_dir / p,
        manifest_path.parent / p,
        artifact_dir / "fp16" / p.name,
        artifact_dir / "fp32" / p.name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def load_embed_weight(
    artifact_dir: Path, embed_name: str, dtype_np: np.dtype
) -> Optional[np.ndarray]:
    try:
        import onnx
        from onnx import numpy_helper
    except Exception:
        return None

    try:
        embed_path = artifact_dir / embed_name
        if not embed_path.exists():
            alt16 = artifact_dir / "fp16" / Path(embed_name).name
            alt32 = artifact_dir / "fp32" / Path(embed_name).name
            if alt16.exists():
                embed_path = alt16
            elif alt32.exists():
                embed_path = alt32
        m = onnx.load(str(embed_path.resolve()), load_external_data=True)
        if not m.graph.initializer:
            return None
        # Fast lookup is only valid when the ONNX embed graph exposes a direct
        # [vocab, hidden] floating-point table. Quantized graphs often rewrite
        # Gather inputs and add dequant nodes, so fall back to ONNX session path.
        best = None
        best_elems = -1
        for init in m.graph.initializer:
            try:
                arr = numpy_helper.to_array(init)
            except Exception:
                continue
            if arr is None or arr.ndim != 2:
                continue
            if arr.dtype.kind != "f":
                continue
            elems = int(arr.size)
            if elems > best_elems:
                best = arr
                best_elems = elems
        if best is None:
            return None
        return np.asarray(best, dtype=dtype_np)
    except Exception:
        return None


def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in s.split("\n")).strip()


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    m = float(np.max(x))
    y = np.exp(x - m)
    z = float(np.sum(y))
    if not np.isfinite(z) or z <= 0:
        return np.full_like(y, 1.0 / max(1, y.size), dtype=np.float32)
    return (y / z).astype(np.float32, copy=False)


def sample_next_id(
    logits: np.ndarray,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    rng: np.random.Generator,
) -> int:
    if (not do_sample) or temperature <= 0:
        return int(np.argmax(logits))

    scaled = logits.astype(np.float32, copy=True) / float(max(1e-6, temperature))
    probs = softmax_np(scaled)

    if top_k and top_k > 0 and top_k < probs.size:
        idx = np.argpartition(probs, -top_k)[-top_k:]
        mask = np.zeros_like(probs, dtype=bool)
        mask[idx] = True
        probs = np.where(mask, probs, 0.0)

    if top_p and 0.0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        p_sorted = probs[order]
        csum = np.cumsum(p_sorted)
        keep = csum <= float(top_p)
        if not np.any(keep):
            keep[0] = True
        cut = int(np.sum(keep))
        keep_idx = order[:cut]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep_idx] = True
        probs = np.where(mask, probs, 0.0)

    s = float(np.sum(probs))
    if (not np.isfinite(s)) or s <= 0:
        return int(np.argmax(logits))
    probs = probs / s
    return int(rng.choice(np.arange(probs.size, dtype=np.int64), p=probs))


def quality_score(text: str) -> float:
    t = normalize_text(text)
    if not t:
        return -1e9
    n = len(t)
    unique = len(set(t))
    rep_penalty = 0.0
    if n > 0:
        rep_penalty = (1.0 - (unique / max(1, n))) * 0.5
    trunc_penalty = 0.0
    if t.endswith("<") or t.endswith("&") or t.endswith("```"):
        trunc_penalty += 0.2
    if t.count("<table") != t.count("</table>"):
        trunc_penalty += 0.3
    if t.count("```") % 2 == 1:
        trunc_penalty += 0.2
    return math.log(max(1, n)) - rep_penalty - trunc_penalty


def force_image_size(img: Image.Image, w: int, h: int) -> Image.Image:
    if img.size == (w, h):
        return img
    src_w, src_h = img.size
    scale = min(w / max(1, src_w), h / max(1, src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = img.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (w, h), color=(255, 255, 255))
    x = (w - new_w) // 2
    y = (h - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas


def right_pad_to_len(
    input_ids: np.ndarray, attention_mask: np.ndarray, pad_id: int, target_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    cur = int(input_ids.shape[1])
    if target_len < cur:
        raise RuntimeError(
            f"max_seq_len={target_len} is smaller than prompt seq_len={cur}."
        )
    if target_len == cur:
        return input_ids, attention_mask
    pad_len = target_len - cur
    pad_ids = np.full((1, pad_len), pad_id, dtype=input_ids.dtype)
    pad_mask = np.zeros((1, pad_len), dtype=attention_mask.dtype)
    return (
        np.concatenate([input_ids, pad_ids], axis=1),
        np.concatenate([attention_mask, pad_mask], axis=1),
    )


def build_processor(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "right"
    img_proc = Glm46VImageProcessor.from_pretrained(model_id)
    vid_proc = Glm46VVideoProcessor.from_pretrained(model_id)
    processor = Glm46VProcessor(
        image_processor=img_proc,
        tokenizer=tokenizer,
        video_processor=vid_proc,
    )
    return tokenizer, processor


def build_prompt(tokenizer, prompt_text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def ort_providers(
    device: str,
    cuda_no_fallback: bool = False,
    cuda_ep_tuned: bool = True,
) -> List[Any]:
    device = device.lower().strip()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device in {"cuda", "gpu"}:
        cuda_provider: Any = "CUDAExecutionProvider"
        if cuda_ep_tuned:
            cuda_opts = {"cudnn_conv_use_max_workspace": "1"}
            cuda_provider = ("CUDAExecutionProvider", cuda_opts)
        if cuda_no_fallback:
            return [cuda_provider]
        return [cuda_provider, "CPUExecutionProvider"]
    raise ValueError("device must be cpu or cuda")


def ort_session_options() -> ort.SessionOptions:
    so = ort.SessionOptions()
    # CUDA emits repeated ScatterND warnings in KV decode loops; keep logs actionable.
    so.log_severity_level = 3
    return so


def map_label_to_task(label: str) -> str:
    lab = (label or "").strip().lower()
    for task, labels in LABEL_TASK_MAPPING.items():
        if lab in labels:
            return task
    if "table" in lab:
        return "table"
    if "formula" in lab:
        return "formula"
    return "text"


def get_rope_graph_name(
    cfg: Dict[str, Any],
    prompt_text: str,
    task: str,
    rope_profile: str,
    allow_prompt_override: bool,
) -> str:
    rope_graph_name = cfg["graphs"]["rope"]
    prompt_profiles = cfg.get("prompt_profiles", None)
    if not prompt_profiles:
        return rope_graph_name

    if rope_profile:
        if rope_profile not in prompt_profiles:
            raise RuntimeError(
                f"Unknown --rope_profile '{rope_profile}'. "
                f"Available: {', '.join(prompt_profiles.keys())}"
            )
        return prompt_profiles[rope_profile]["rope"]

    if task in prompt_profiles and prompt_text == PROMPT_PRESETS.get(task, ""):
        return prompt_profiles[task]["rope"]

    for name, info in prompt_profiles.items():
        if prompt_text == info.get("prompt", ""):
            return info["rope"]

    default_profile = cfg.get("default_profile", "document")
    if not allow_prompt_override:
        raise RuntimeError(
            "Prompt does not match any exported rope profile. "
            f"Available profiles: {', '.join(prompt_profiles.keys())}. "
            "Use --rope_profile to select one, or --allow_prompt_override."
        )
    if default_profile not in prompt_profiles:
        default_profile = next(iter(prompt_profiles.keys()))
    return prompt_profiles[default_profile]["rope"]


def crop_with_padding(img: Image.Image, bbox: List[int], pad: int) -> Image.Image:
    if len(bbox) != 4:
        raise RuntimeError(f"bbox_2d must have 4 numbers, got {bbox}")
    w, h = img.size
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1 - pad, w - 1))
    y1 = max(0, min(y1 - pad, h - 1))
    x2 = max(x1 + 1, min(x2 + pad, w))
    y2 = max(y1 + 1, min(y2 + pad, h))
    return img.crop((x1, y1, x2, y2))


def flatten_regions(layout_json: Any, page_index: int) -> List[Dict[str, Any]]:
    # Expected shape: [ [region, ...], [region, ...], ... ] (per-page list)
    if (
        isinstance(layout_json, list)
        and layout_json
        and all(isinstance(x, list) for x in layout_json)
    ):
        if page_index < 0 or page_index >= len(layout_json):
            raise RuntimeError(
                f"--layout_page_index={page_index} out of range for {len(layout_json)} pages."
            )
        return [r for r in layout_json[page_index] if isinstance(r, dict)]

    # Fallback: single-page flat list
    if isinstance(layout_json, list):
        return [r for r in layout_json if isinstance(r, dict)]
    raise RuntimeError("Unsupported layout json format.")


def run_single_ocr(
    *,
    cfg: Dict[str, Any],
    tokenizer,
    processor,
    sess_v,
    sess_e,
    sess_r,
    sess_d,
    sess_dp=None,
    sess_ds=None,
    embed_weight: Optional[np.ndarray],
    pil_image: Image.Image,
    prompt_text: str,
    max_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    **_unused,
) -> Tuple[str, Dict[str, Any]]:
    image_w = int(cfg["image_size"]["width"])
    image_h = int(cfg["image_size"]["height"])
    max_seq_len = int(cfg["max_seq_len"])
    t_img = int(cfg["t_img"])
    image_token_id = int(cfg["image_token_id"])
    dtype_np = np.float16 if cfg["dtype"] == "float16" else np.float32

    prompt = build_prompt(tokenizer, prompt_text)
    img = force_image_size(pil_image, image_w, image_h)
    inputs = processor(
        text=prompt,
        images=img,
        return_tensors="np",
        images_kwargs={"do_resize": False},
    )
    input_ids, attention_mask = right_pad_to_len(
        inputs["input_ids"],
        inputs["attention_mask"],
        tokenizer.pad_token_id,
        max_seq_len,
    )
    attention_mask = attention_mask.astype(np.int64, copy=False)
    pixel_values = inputs["pixel_values"].astype(dtype_np, copy=False)
    image_grid_thw = inputs["image_grid_thw"].astype(np.int64, copy=False)

    n_img_tokens = int((input_ids == image_token_id).sum())
    if n_img_tokens != t_img:
        raise RuntimeError(
            f"Expected {t_img} image tokens for static profile, got {n_img_tokens}."
        )
    idxs = np.where(input_ids[0] == image_token_id)[0]
    img_start = int(idxs.min())
    img_end_excl = img_start + t_img
    if not np.array_equal(idxs, np.arange(img_start, img_end_excl)):
        raise RuntimeError("Image token block is not contiguous.")

    prompt_len = int(attention_mask[0].sum())
    eos_ids_cfg = cfg.get("eos_token_ids", None)
    if eos_ids_cfg is None:
        tok_eos = tokenizer.eos_token_id
        eos_ids = {int(tok_eos)} if tok_eos is not None else set()
    elif isinstance(eos_ids_cfg, int):
        eos_ids = {int(eos_ids_cfg)}
    else:
        eos_ids = {int(x) for x in eos_ids_cfg}
    pad_id = tokenizer.pad_token_id

    if sess_d is None:
        raise RuntimeError("decode session is required for non-KV decoding.")

    image_embeds = sess_v.run(
        ["image_embeds"],
        {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
    )[0].astype(dtype_np)

    if embed_weight is not None:
        token_embeds = embed_weight[input_ids]
    else:
        token_embeds = None
    spliced = None
    embed_token_cache: Optional[Dict[int, np.ndarray]] = None
    tmp_ids = None

    pos_ids, _ = sess_r.run(
        ["position_ids", "rope_deltas"],
        {
            "input_ids": input_ids,
            "image_grid_thw": image_grid_thw,
            "attention_mask": attention_mask,
        },
    )
    pos_ids = pos_ids.astype(np.int64)

    write_pos = prompt_len
    next_text_pos = pos_ids[:, 0, prompt_len - 1].copy() + 1
    fast_greedy = (
        (not do_sample)
        and (float(repetition_penalty) <= 1.0)
        and (int(no_repeat_ngram_size) <= 1)
    )
    rng = None if fast_greedy else np.random.default_rng(int(seed))
    if token_embeds is None:
        if sess_e is None:
            raise RuntimeError(
                "embed session is required when embed weight is unavailable."
            )
        token_embeds = sess_e.run(["token_embeds"], {"input_ids": input_ids})[
            0
        ].astype(dtype_np)
        tmp_ids = np.full((1, max_seq_len), pad_id, dtype=np.int64)
        embed_token_cache = {}

    ended_by_eos = False
    seen_token_ids = (
        set(int(t) for t in input_ids[0, :write_pos].tolist()) if not fast_greedy else None
    )
    for _ in range(int(max_new_tokens)):
        if write_pos >= input_ids.shape[1]:
            break
        pos_ids[:, 0, write_pos] = next_text_pos

        if spliced is None:
            spliced = token_embeds.copy()
        else:
            # Keep a stable allocation to reduce per-step memory churn.
            np.copyto(spliced, token_embeds)
        spliced[:, img_start:img_end_excl, :] = image_embeds

        logits = sess_d.run(
            ["logits"],
            {
                "inputs_embeds": spliced,
                "attention_mask": attention_mask,
                "position_ids": pos_ids,
            },
        )[0]
        if fast_greedy:
            next_id = int(np.argmax(logits[0, 0]))
        else:
            next_logits = logits[0, 0].astype(np.float32, copy=True)

            if repetition_penalty and repetition_penalty > 1.0:
                for tid in seen_token_ids:
                    v = next_logits[int(tid)]
                    if v < 0:
                        next_logits[int(tid)] = v * float(repetition_penalty)
                    else:
                        next_logits[int(tid)] = v / float(repetition_penalty)

            n = int(no_repeat_ngram_size)
            if n > 1 and write_pos >= (n - 1):
                seq = input_ids[0, :write_pos].tolist()
                prefix = tuple(seq[-(n - 1) :])
                banned = []
                end = len(seq) - n + 1
                for i in range(max(0, end)):
                    gram = tuple(seq[i : i + n])
                    if gram[:-1] == prefix:
                        banned.append(gram[-1])
                if banned:
                    next_logits[np.array(banned, dtype=np.int64)] = -1e30

            next_id = sample_next_id(
                next_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                rng=rng,
            )

        cur_pos = write_pos
        input_ids[0, write_pos] = next_id
        attention_mask[0, write_pos] = 1
        if seen_token_ids is not None:
            seen_token_ids.add(int(next_id))
        write_pos += 1
        next_text_pos += 1
        if embed_weight is not None:
            token_embeds[0, cur_pos, :] = embed_weight[next_id]
            if spliced is not None and not (img_start <= cur_pos < img_end_excl):
                spliced[0, cur_pos, :] = token_embeds[0, cur_pos, :]
        else:
            if embed_token_cache is None:
                embed_token_cache = {}
            cached = embed_token_cache.get(int(next_id))
            if cached is None:
                if sess_e is None or tmp_ids is None:
                    raise RuntimeError(
                        "embed session is required when embed weight is unavailable."
                    )
                tmp_ids[0, :] = pad_id
                tmp_ids[0, 0] = next_id
                one = sess_e.run(["token_embeds"], {"input_ids": tmp_ids})[0][
                    0, 0
                ].astype(dtype_np, copy=True)
                embed_token_cache[int(next_id)] = one
                cached = one
            token_embeds[0, cur_pos, :] = cached
            if spliced is not None and not (img_start <= cur_pos < img_end_excl):
                spliced[0, cur_pos, :] = cached

        if next_id in eos_ids:
            ended_by_eos = True
            break

    real_len = int(attention_mask[0].sum())
    new_ids = input_ids[0, prompt_len:real_len].tolist()
    new_ids = [tid for tid in new_ids if tid not in (image_token_id, pad_id)]
    text = normalize_text(tokenizer.decode(new_ids, skip_special_tokens=True))
    meta = {
        "generated_tokens": int(len(new_ids)),
        "ended_by_eos": bool(ended_by_eos),
        "hit_token_limit": (not ended_by_eos) and (len(new_ids) >= int(max_new_tokens)),
    }
    return text, meta


def run_single_ocr_kv(
    *,
    cfg: Dict[str, Any],
    tokenizer,
    processor,
    sess_v,
    sess_e,
    sess_r,
    sess_d=None,
    sess_dp,
    sess_ds,
    embed_weight: Optional[np.ndarray],
    pil_image: Image.Image,
    prompt_text: str,
    max_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    **_unused,
) -> Tuple[str, Dict[str, Any]]:
    image_w = int(cfg["image_size"]["width"])
    image_h = int(cfg["image_size"]["height"])
    max_seq_len = int(cfg["max_seq_len"])
    t_img = int(cfg["t_img"])
    image_token_id = int(cfg["image_token_id"])
    dtype_np = np.float16 if cfg["dtype"] == "float16" else np.float32
    kv_meta = cfg.get("kv_cache", {})
    n_layers = int(kv_meta.get("num_layers", 0))
    if n_layers <= 0:
        raise RuntimeError("kv_cache metadata is missing from manifest.")
    if sess_dp is None or sess_ds is None:
        raise RuntimeError("KV decode sessions are required for run_single_ocr_kv.")
    kv_iobinding = bool(_unused.get("kv_iobinding", False))

    prompt = build_prompt(tokenizer, prompt_text)
    img = force_image_size(pil_image, image_w, image_h)
    inputs = processor(
        text=prompt,
        images=img,
        return_tensors="np",
        images_kwargs={"do_resize": False},
    )
    input_ids, attention_mask = right_pad_to_len(
        inputs["input_ids"],
        inputs["attention_mask"],
        tokenizer.pad_token_id,
        max_seq_len,
    )
    attention_mask = attention_mask.astype(np.int64, copy=False)
    pixel_values = inputs["pixel_values"].astype(dtype_np, copy=False)
    image_grid_thw = inputs["image_grid_thw"].astype(np.int64, copy=False)

    n_img_tokens = int((input_ids == image_token_id).sum())
    if n_img_tokens != t_img:
        raise RuntimeError(
            f"Expected {t_img} image tokens for static profile, got {n_img_tokens}."
        )
    idxs = np.where(input_ids[0] == image_token_id)[0]
    img_start = int(idxs.min())
    img_end_excl = img_start + t_img
    if not np.array_equal(idxs, np.arange(img_start, img_end_excl)):
        raise RuntimeError("Image token block is not contiguous.")

    prompt_len = int(attention_mask[0].sum())
    eos_ids_cfg = cfg.get("eos_token_ids", None)
    if eos_ids_cfg is None:
        tok_eos = tokenizer.eos_token_id
        eos_ids = {int(tok_eos)} if tok_eos is not None else set()
    elif isinstance(eos_ids_cfg, int):
        eos_ids = {int(eos_ids_cfg)}
    else:
        eos_ids = {int(x) for x in eos_ids_cfg}
    pad_id = tokenizer.pad_token_id

    image_embeds = sess_v.run(
        ["image_embeds"],
        {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
    )[0].astype(dtype_np)

    if embed_weight is not None:
        token_embeds = embed_weight[input_ids]
    else:
        if sess_e is None:
            raise RuntimeError(
                "embed session is required when embed weight is unavailable."
            )
        token_embeds = sess_e.run(["token_embeds"], {"input_ids": input_ids})[0].astype(
            dtype_np
        )
    spliced = token_embeds.copy()
    spliced[:, img_start:img_end_excl, :] = image_embeds

    pos_ids, _ = sess_r.run(
        ["position_ids", "rope_deltas"],
        {
            "input_ids": input_ids,
            "image_grid_thw": image_grid_thw,
            "attention_mask": attention_mask,
        },
    )
    pos_ids = pos_ids.astype(np.int64)
    next_text_pos = pos_ids[:, 0, prompt_len - 1].copy() + 1
    if prompt_len < max_seq_len:
        pos_ids[:, 0, prompt_len] = next_text_pos

    prefill_out_names = ["logits"] + [
        x for i in range(n_layers) for x in (f"past_key_{i}", f"past_value_{i}")
    ]
    attention_mask_i64 = attention_mask.astype(np.int64, copy=False)
    pos_ids_i64 = pos_ids.astype(np.int64, copy=False)
    fast_greedy = (
        (not do_sample)
        and (float(repetition_penalty) <= 1.0)
        and (int(no_repeat_ngram_size) <= 1)
    )
    past = None
    past_ov = None
    if kv_iobinding:
        prefill_io = sess_dp.io_binding()
        prefill_io.bind_cpu_input("inputs_embeds", spliced.astype(dtype_np, copy=False))
        prefill_io.bind_cpu_input("attention_mask", attention_mask_i64)
        prefill_io.bind_cpu_input("position_ids", pos_ids_i64)
        for out_name in prefill_out_names:
            prefill_io.bind_output(out_name, "cuda")
        sess_dp.run_with_iobinding(prefill_io)
        prefill_ov = prefill_io.get_outputs()
        logits0 = prefill_ov[0].numpy()[0, 0]
        next_logits = logits0 if fast_greedy else logits0.astype(np.float32, copy=True)
        past_ov = list(prefill_ov[1:])
    else:
        prefill = sess_dp.run(
            prefill_out_names,
            {
                "inputs_embeds": spliced.astype(dtype_np, copy=False),
                "attention_mask": attention_mask_i64,
                "position_ids": pos_ids_i64,
            },
        )
        next_logits = (
            prefill[0][0, 0]
            if fast_greedy
            else prefill[0][0, 0].astype(np.float32, copy=True)
        )
        past = list(prefill[1:])

    write_pos = prompt_len
    rng = None if fast_greedy else np.random.default_rng(int(seed))
    ended_by_eos = False
    generated = 0
    seen_token_ids = (
        set(int(t) for t in input_ids[0, :write_pos].tolist()) if not fast_greedy else None
    )
    step_out_names = ["logits"] + [
        x for i in range(n_layers) for x in (f"present_key_{i}", f"present_value_{i}")
    ]
    past_key_names = [f"past_key_{i}" for i in range(n_layers)]
    past_value_names = [f"past_value_{i}" for i in range(n_layers)]
    step_pos = np.empty((3, 1, 1), dtype=np.int64)
    cache_position = np.empty((1,), dtype=np.int64)
    step_feed = {
        "attention_mask": attention_mask_i64,
        "position_ids": step_pos,
        "cache_position": cache_position,
    }
    tmp_ids = None
    embed_token_cache: Optional[Dict[int, np.ndarray]] = None
    if embed_weight is None:
        tmp_ids = np.full((1, max_seq_len), pad_id, dtype=np.int64)
        embed_token_cache = {}

    for _ in range(int(max_new_tokens)):
        if write_pos >= input_ids.shape[1]:
            break

        if fast_greedy:
            next_id = int(np.argmax(next_logits))
        else:
            if repetition_penalty and repetition_penalty > 1.0:
                for tid in seen_token_ids:
                    v = next_logits[int(tid)]
                    if v < 0:
                        next_logits[int(tid)] = v * float(repetition_penalty)
                    else:
                        next_logits[int(tid)] = v / float(repetition_penalty)

            n = int(no_repeat_ngram_size)
            if n > 1 and write_pos >= (n - 1):
                seq = input_ids[0, :write_pos].tolist()
                prefix = tuple(seq[-(n - 1) :])
                banned = []
                end = len(seq) - n + 1
                for i in range(max(0, end)):
                    gram = tuple(seq[i : i + n])
                    if gram[:-1] == prefix:
                        banned.append(gram[-1])
                if banned:
                    next_logits[np.array(banned, dtype=np.int64)] = -1e30

            next_id = sample_next_id(
                next_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                rng=rng,
            )

        input_ids[0, write_pos] = next_id
        attention_mask[0, write_pos] = 1
        if seen_token_ids is not None:
            seen_token_ids.add(int(next_id))
        generated += 1
        if next_id in eos_ids:
            ended_by_eos = True
            break

        # Decode one-step with KV cache.
        if embed_weight is None:
            if sess_e is None:
                raise RuntimeError(
                    "embed session is required when embed weight is unavailable."
                )
            if embed_token_cache is None:
                embed_token_cache = {}
            cached = embed_token_cache.get(int(next_id))
            if cached is None:
                tmp_ids[0, :] = pad_id
                tmp_ids[0, 0] = next_id
                one = sess_e.run(["token_embeds"], {"input_ids": tmp_ids})[0][
                    0, 0
                ].astype(dtype_np, copy=True)
                embed_token_cache[int(next_id)] = one
                cached = one
            cur_emb = cached[None, None, :]
        else:
            cur_emb = embed_weight[int(next_id)][None, None, :].astype(
                dtype_np, copy=False
            )
        cache_position[0] = write_pos
        step_pos[:, 0, 0] = next_text_pos
        next_text_pos += 1

        if kv_iobinding:
            cur_emb_ov = ort.OrtValue.ortvalue_from_numpy(cur_emb, "cuda", 0)
            step_io = sess_ds.io_binding()
            step_io.bind_ortvalue_input("cur_emb", cur_emb_ov)
            step_io.bind_cpu_input("attention_mask", attention_mask_i64)
            step_io.bind_cpu_input("position_ids", step_pos)
            step_io.bind_cpu_input("cache_position", cache_position)
            for i in range(n_layers):
                step_io.bind_ortvalue_input(past_key_names[i], past_ov[2 * i])
                step_io.bind_ortvalue_input(past_value_names[i], past_ov[2 * i + 1])
            for out_name in step_out_names:
                step_io.bind_output(out_name, "cuda")
            sess_ds.run_with_iobinding(step_io)
            step_ov = step_io.get_outputs()
            logits_step = step_ov[0].numpy()[0, 0]
            next_logits = (
                logits_step
                if fast_greedy
                else logits_step.astype(np.float32, copy=True)
            )
            past_ov = list(step_ov[1:])
        else:
            step_feed["cur_emb"] = cur_emb
            for i in range(n_layers):
                step_feed[past_key_names[i]] = past[2 * i]
                step_feed[past_value_names[i]] = past[2 * i + 1]
            step_out = sess_ds.run(step_out_names, step_feed)
            next_logits = (
                step_out[0][0, 0]
                if fast_greedy
                else step_out[0][0, 0].astype(np.float32, copy=True)
            )
            past = list(step_out[1:])
        write_pos += 1

    real_len = int(attention_mask[0].sum())
    new_ids = input_ids[0, prompt_len:real_len].tolist()
    new_ids = [tid for tid in new_ids if tid not in (image_token_id, pad_id)]
    text = normalize_text(tokenizer.decode(new_ids, skip_special_tokens=True))
    meta = {
        "generated_tokens": int(len(new_ids)),
        "ended_by_eos": bool(ended_by_eos),
        "hit_token_limit": (not ended_by_eos) and (generated >= int(max_new_tokens)),
    }
    return text, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument(
        "--task", choices=["document", "text", "table", "formula"], default="document"
    )
    ap.add_argument("--prompt", default="")
    ap.add_argument("--rope_profile", default="")
    ap.add_argument("--allow_prompt_override", action="store_true")
    ap.add_argument("--layout_json", default="")
    ap.add_argument("--layout_page_index", type=int, default=0)
    ap.add_argument("--region_pad", type=int, default=12)
    ap.add_argument("--layout_out_json", default="")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_new_tokens_text", type=int, default=0)
    ap.add_argument("--max_new_tokens_table", type=int, default=0)
    ap.add_argument("--max_new_tokens_formula", type=int, default=0)
    ap.add_argument("--max_new_tokens_document", type=int, default=0)
    ap.add_argument("--adaptive_budgets", default="")
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--num_candidates", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--official_quality", action="store_true")
    ap.add_argument("--fast_embed_lookup", action="store_true")
    ap.add_argument("--disable_kv_cache", action="store_true")
    ap.add_argument(
        "--vision_policy",
        default="fp16",
        choices=["auto", "fp16", "quant", "table_quant"],
        help=(
            "Vision session selection when manifest provides graphs.vision_quant. "
            "auto: text->fp16, others->quant; fp16: always graphs.vision; "
            "quant: always graphs.vision_quant (if present); "
            "table_quant: quant only for table task."
        ),
    )
    ap.add_argument(
        "--disable_kv_iobinding",
        action="store_true",
        help="Disable CUDA KV I/O binding and use numpy-based KV transfers.",
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--cuda_no_fallback",
        action="store_true",
        help="Use only CUDAExecutionProvider when --device=cuda (no CPU fallback partition).",
    )
    ap.add_argument(
        "--disable_cuda_ep_tuning",
        action="store_true",
        help="Disable CUDA EP perf options (e.g., cudnn_conv_use_max_workspace).",
    )
    ap.add_argument("--out_text", default="")
    args = ap.parse_args()
    if args.official_quality:
        # OCR quality is more stable with deterministic decoding.
        args.do_sample = False
        args.temperature = 1.0
        args.top_p = 1.0
        args.top_k = 0
        args.repetition_penalty = 1.0
        if args.max_new_tokens < 1024:
            args.max_new_tokens = 1024
        args.num_candidates = 1

    def per_task_budget(task: str) -> int:
        if task == "text" and args.max_new_tokens_text > 0:
            return int(args.max_new_tokens_text)
        if task == "table" and args.max_new_tokens_table > 0:
            return int(args.max_new_tokens_table)
        if task == "formula" and args.max_new_tokens_formula > 0:
            return int(args.max_new_tokens_formula)
        if task == "document" and args.max_new_tokens_document > 0:
            return int(args.max_new_tokens_document)
        return int(args.max_new_tokens)

    def budgets_for_task(task: str) -> List[int]:
        base = max(1, per_task_budget(task))
        vals = [base]
        if args.adaptive_budgets.strip():
            for tok in args.adaptive_budgets.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    v = int(tok)
                except ValueError:
                    continue
                if v > 0:
                    vals.append(v)
        elif args.official_quality:
            # Allow one larger deterministic retry when the first pass is token-limited.
            vals.append(base * 2)
        elif task in {"table", "formula"} and not args.official_quality:
            # Structured outputs frequently need more tokens than generic text;
            # escalate only when not explicitly overridden.
            vals.append(base * 2)
        vals = sorted(set(vals))
        return vals

    artifact_dir = Path(args.artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path = artifact_dir.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {artifact_dir} or its parent."
        )
    cfg = read_json(manifest_path)

    model_id = cfg["model_id"]
    tokenizer, processor = build_processor(model_id)

    providers = ort_providers(
        args.device,
        cuda_no_fallback=args.cuda_no_fallback,
        cuda_ep_tuned=(not args.disable_cuda_ep_tuning),
    )
    sess_options = ort_session_options()
    graphs = cfg["graphs"]
    vision_path = resolve_graph_path(artifact_dir, manifest_path, graphs["vision"])
    embed_path = resolve_graph_path(artifact_dir, manifest_path, graphs["embed"])
    sess_v = None
    vision_q_path = None
    sess_v_quant = None
    if "vision_quant" in graphs:
        vision_q_path = resolve_graph_path(
            artifact_dir, manifest_path, graphs["vision_quant"]
        )
    sess_e = None
    sess_d = None
    sess_dp = None
    sess_ds = None
    use_kv_cache = (
        (not args.disable_kv_cache)
        and ("decode_prefill_kv" in graphs)
        and ("decode_step_kv" in graphs)
        and ("kv_cache" in cfg)
    )
    use_kv_iobinding = bool(use_kv_cache and args.device == "cuda" and (not args.disable_kv_iobinding))
    if use_kv_cache:
        dp_path = resolve_graph_path(
            artifact_dir, manifest_path, graphs["decode_prefill_kv"]
        )
        ds_path = resolve_graph_path(
            artifact_dir, manifest_path, graphs["decode_step_kv"]
        )
        sess_dp = ort.InferenceSession(
            str(dp_path.resolve()),
            sess_options=sess_options,
            providers=providers,
        )
        sess_ds = ort.InferenceSession(
            str(ds_path.resolve()),
            sess_options=sess_options,
            providers=providers,
        )
    dtype_np = np.float16 if cfg["dtype"] == "float16" else np.float32
    embed_weight = None
    # KV decode benefits heavily from direct embedding lookup to avoid
    # per-token static-shape embed ONNX calls.
    if args.fast_embed_lookup or use_kv_cache:
        embed_weight = load_embed_weight(artifact_dir, graphs["embed"], dtype_np)
    if embed_weight is None:
        sess_e = ort.InferenceSession(
            str(embed_path.resolve()),
            sess_options=sess_options,
            providers=providers,
        )
    if not use_kv_cache:
        decode_path = resolve_graph_path(artifact_dir, manifest_path, graphs["decode"])
        sess_d = ort.InferenceSession(
            str(decode_path.resolve()),
            sess_options=sess_options,
            providers=providers,
        )
    rope_sessions: Dict[str, ort.InferenceSession] = {}

    def get_rope_session(rope_name: str) -> ort.InferenceSession:
        sess = rope_sessions.get(rope_name)
        if sess is None:
            sess = ort.InferenceSession(
                str(
                    resolve_graph_path(artifact_dir, manifest_path, rope_name).resolve()
                ),
                sess_options=sess_options,
                providers=providers,
            )
            rope_sessions[rope_name] = sess
        return sess

    src_img = Image.open(args.image).convert("RGB")

    def select_vision_session(task: str):
        nonlocal sess_v, sess_v_quant
        def get_sess_v():
            nonlocal sess_v
            if sess_v is None:
                sess_v = ort.InferenceSession(
                    str(vision_path.resolve()),
                    sess_options=sess_options,
                    providers=providers,
                )
            return sess_v

        if vision_q_path is None:
            return get_sess_v()

        def get_sess_v_quant():
            nonlocal sess_v_quant
            if sess_v_quant is None:
                sess_v_quant = ort.InferenceSession(
                    str(vision_q_path.resolve()),
                    sess_options=sess_options,
                    providers=providers,
                )
            return sess_v_quant

        pol = args.vision_policy
        if pol == "fp16":
            return get_sess_v()
        if pol == "quant":
            return get_sess_v_quant()
        if pol == "table_quant":
            if task == "table":
                return get_sess_v_quant()
            return get_sess_v()
        # auto
        if task == "text":
            return get_sess_v()
        return get_sess_v_quant()

    if args.layout_json:
        layout_obj = read_json(Path(args.layout_json))
        regions = flatten_regions(layout_obj, args.layout_page_index)
        # Sort by reading order: top-to-bottom, then left-to-right
        regions = sorted(
            [r for r in regions if isinstance(r.get("bbox_2d"), list)],
            key=lambda r: (int(r["bbox_2d"][1]), int(r["bbox_2d"][0])),
        )
        outputs = []
        out_regions = []
        for idx, region in enumerate(regions):
            label = str(region.get("label", "")).strip()
            task = map_label_to_task(label)
            prompt_text = (
                args.prompt
                if args.prompt
                else PROMPT_PRESETS.get(task, PROMPT_PRESETS["document"])
            )
            rope_graph_name = get_rope_graph_name(
                cfg,
                prompt_text=prompt_text,
                task=task,
                rope_profile=args.rope_profile,
                allow_prompt_override=args.allow_prompt_override,
            )
            sess_r = get_rope_session(rope_graph_name)
            crop = crop_with_padding(src_img, region["bbox_2d"], args.region_pad)
            best_text = ""
            best_q = -1e18
            for budget in budgets_for_task(task):
                budget_best_text = ""
                budget_best_q = -1e18
                budget_meta = None
                for cidx in range(max(1, int(args.num_candidates))):
                    runner = run_single_ocr_kv if use_kv_cache else run_single_ocr
                    sess_v_sel = select_vision_session(task)
                    text_i, meta_i = runner(
                        cfg=cfg,
                        tokenizer=tokenizer,
                        processor=processor,
                        sess_v=sess_v_sel,
                        sess_e=sess_e,
                        sess_r=sess_r,
                        sess_d=sess_d if not use_kv_cache else None,
                        sess_dp=sess_dp if use_kv_cache else None,
                        sess_ds=sess_ds if use_kv_cache else None,
                        embed_weight=embed_weight,
                        pil_image=crop,
                        prompt_text=prompt_text,
                        max_new_tokens=budget,
                        repetition_penalty=args.repetition_penalty,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        do_sample=(args.do_sample and task not in {"table", "formula"}),
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        seed=args.seed + cidx + idx * 9973 + budget * 17,
                        kv_iobinding=use_kv_iobinding,
                    )
                    q = quality_score(text_i)
                    if q > budget_best_q:
                        budget_best_q = q
                        budget_best_text = text_i
                        budget_meta = meta_i
                if budget_best_q > best_q:
                    best_q = budget_best_q
                    best_text = budget_best_text
                # Escalate only when output appears truncated by token budget.
                if not (budget_meta and budget_meta.get("hit_token_limit")):
                    break
            text_i = best_text
            if text_i:
                outputs.append(text_i)
            out_regions.append(
                {
                    "index": int(region.get("index", idx)),
                    "label": label,
                    "task": task,
                    "bbox_2d": region["bbox_2d"],
                    "content": text_i,
                }
            )
        text = normalize_text("\n\n".join(outputs))
        if args.layout_out_json:
            Path(args.layout_out_json).write_text(
                json.dumps(out_regions, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    else:
        prompt_text = args.prompt if args.prompt else PROMPT_PRESETS[args.task]
        rope_graph_name = get_rope_graph_name(
            cfg,
            prompt_text=prompt_text,
            task=args.task,
            rope_profile=args.rope_profile,
            allow_prompt_override=args.allow_prompt_override,
        )
        sess_r = get_rope_session(rope_graph_name)
        best_text = ""
        best_q = -1e18
        for budget in budgets_for_task(args.task):
            budget_best_text = ""
            budget_best_q = -1e18
            budget_meta = None
            for cidx in range(max(1, int(args.num_candidates))):
                runner = run_single_ocr_kv if use_kv_cache else run_single_ocr
                sess_v_sel = select_vision_session(args.task)
                text_i, meta_i = runner(
                    cfg=cfg,
                    tokenizer=tokenizer,
                    processor=processor,
                    sess_v=sess_v_sel,
                    sess_e=sess_e,
                    sess_r=sess_r,
                    sess_d=sess_d if not use_kv_cache else None,
                    sess_dp=sess_dp if use_kv_cache else None,
                    sess_ds=sess_ds if use_kv_cache else None,
                    embed_weight=embed_weight,
                    pil_image=src_img,
                    prompt_text=prompt_text,
                    max_new_tokens=budget,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    do_sample=(
                        args.do_sample and args.task not in {"table", "formula"}
                    ),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    seed=args.seed + cidx + budget * 17,
                    kv_iobinding=use_kv_iobinding,
                )
                q = quality_score(text_i)
                if q > budget_best_q:
                    budget_best_q = q
                    budget_best_text = text_i
                    budget_meta = meta_i
            if budget_best_q > best_q:
                best_q = budget_best_q
                best_text = budget_best_text
            if not (budget_meta and budget_meta.get("hit_token_limit")):
                break
        text = best_text

    print(text)
    if args.out_text:
        out_path = Path(args.out_text)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
