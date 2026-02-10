#!/usr/bin/env python3
"""Export fixed task presets for browser-side preprocessing.

The output contains pre-tokenized/padded inputs for default task prompts so the
browser runtime can preprocess images without requiring tokenizer execution in JS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoTokenizer, Glm46VImageProcessor, Glm46VProcessor, Glm46VVideoProcessor

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


def build_prompt(tokenizer, prompt_text: str) -> str:
    chat = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        }
    ]
    return tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)


def right_pad(input_ids: np.ndarray, attention_mask: np.ndarray, pad_id: int, target_len: int):
    cur = int(input_ids.shape[1])
    if cur > target_len:
        raise RuntimeError(f"prompt seq_len={cur} exceeds max_seq_len={target_len}")
    if cur == target_len:
        return input_ids, attention_mask
    pad = target_len - cur
    ids = np.full((1, pad), pad_id, dtype=input_ids.dtype)
    m = np.zeros((1, pad), dtype=attention_mask.dtype)
    return np.concatenate([input_ids, ids], axis=1), np.concatenate([attention_mask, m], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="_local/variants/custom_w8_prefill8m_dual_vision/manifest.json")
    ap.add_argument("--out", default="web_runtime/task_presets.json")
    args = ap.parse_args()

    manifest = read_json(Path(args.manifest))
    model_id = manifest["model_id"]
    image_w = int(manifest["image_size"]["width"])
    image_h = int(manifest["image_size"]["height"])
    max_seq_len = int(manifest["max_seq_len"])
    t_img = int(manifest["t_img"])
    hidden_size = int(manifest["hidden_size"])

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "right"
    processor = Glm46VProcessor(
        image_processor=Glm46VImageProcessor.from_pretrained(model_id),
        tokenizer=tokenizer,
        video_processor=Glm46VVideoProcessor.from_pretrained(model_id),
    )
    blank = Image.new("RGB", (image_w, image_h), color=(255, 255, 255))

    tasks = {}
    for task, prompt in PROMPT_PRESETS.items():
        prompt_text = build_prompt(tokenizer, prompt)
        out = processor(
            text=prompt_text,
            images=blank,
            return_tensors="np",
            images_kwargs={"do_resize": False},
        )
        input_ids, attention_mask = right_pad(
            out["input_ids"], out["attention_mask"], tokenizer.pad_token_id, max_seq_len
        )
        tasks[task] = {
            "task": task,
            "prompt": prompt,
            "input_ids": input_ids[0].astype(np.int32).tolist(),
            "attention_mask": attention_mask[0].astype(np.int32).tolist(),
            "image_grid_thw": out["image_grid_thw"][0].astype(np.int64).tolist(),
            "seq_len": int(out["attention_mask"][0].sum()),
        }

    payload = {
        "model_id": model_id,
        "max_seq_len": max_seq_len,
        "t_img": t_img,
        "hidden_size": hidden_size,
        "image_size": {"width": image_w, "height": image_h},
        "tasks": tasks,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()

