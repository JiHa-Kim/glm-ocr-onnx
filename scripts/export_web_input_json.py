#!/usr/bin/env python3
"""Export browser demo input JSON for web_runtime/index.html."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import (
    AutoTokenizer,
    Glm46VImageProcessor,
    Glm46VProcessor,
    Glm46VVideoProcessor,
)


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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_image(img: Image.Image, w: int, h: int) -> Image.Image:
    if img.size == (w, h):
        return img
    src_w, src_h = img.size
    scale = min(w / max(1, src_w), h / max(1, src_h))
    nw = max(1, int(round(src_w * scale)))
    nh = max(1, int(round(src_h * scale)))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(resized, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def right_pad(input_ids: np.ndarray, attention_mask: np.ndarray, pad_id: int, target: int) -> tuple[np.ndarray, np.ndarray]:
    cur = int(input_ids.shape[1])
    if cur > target:
        raise RuntimeError(f"prompt seq_len={cur} exceeds max_seq_len={target}")
    if cur == target:
        return input_ids, attention_mask
    pad = target - cur
    ids = np.full((1, pad), pad_id, dtype=input_ids.dtype)
    m = np.zeros((1, pad), dtype=attention_mask.dtype)
    return np.concatenate([input_ids, ids], axis=1), np.concatenate([attention_mask, m], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifact_glm_ocr_web_split/manifest.json")
    ap.add_argument("--image", required=True)
    ap.add_argument("--task", choices=["document", "text", "table", "formula"], default="document")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--out", default="web_runtime/demo_input.json")
    args = ap.parse_args()

    manifest = read_json(Path(args.manifest))
    model_id = manifest["model_id"]
    max_seq = int(manifest["max_seq_len"])
    image_w = int(manifest["image_size"]["width"])
    image_h = int(manifest["image_size"]["height"])
    t_img = int(manifest["t_img"])
    hidden = int(manifest["hidden_size"])

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "right"
    processor = Glm46VProcessor(
        image_processor=Glm46VImageProcessor.from_pretrained(model_id),
        tokenizer=tokenizer,
        video_processor=Glm46VVideoProcessor.from_pretrained(model_id),
    )

    prompt_text = args.prompt if args.prompt else PROMPT_PRESETS[args.task]
    chat = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        }
    ]
    prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    img = normalize_image(Image.open(args.image).convert("RGB"), image_w, image_h)
    out = processor(
        text=prompt,
        images=img,
        return_tensors="np",
        images_kwargs={"do_resize": False},
    )
    input_ids, attention_mask = right_pad(
        out["input_ids"], out["attention_mask"], tokenizer.pad_token_id, max_seq
    )
    pixel_values = np.asarray(out["pixel_values"], dtype=np.float32)
    grid = np.asarray(out["image_grid_thw"], dtype=np.int64)
    seq_len = int(attention_mask[0].sum())

    payload = {
        "input_ids": input_ids[0].astype(np.int32).tolist(),
        "attention_mask": attention_mask[0].astype(np.int32).tolist(),
        "image_grid_thw": grid[0].astype(np.int64).tolist(),
        "pixel_values": pixel_values.reshape(-1).astype(np.float32).tolist(),
        "seq_len": seq_len,
        "t_img": t_img,
        "hidden_size": hidden,
        "max_new_tokens": int(args.max_new_tokens),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
