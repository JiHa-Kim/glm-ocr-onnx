import type { ModelInputs, TaskPresetEntry, TaskPresetFile, WebRuntimeManifest } from "./types.js";

const IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073];
const IMAGE_STD = [0.26862954, 0.26130258, 0.27577711];
const PATCH_SIZE = 14;
const TEMPORAL_PATCH_SIZE = 2;
const MERGE_SIZE = 2;

function toI32(xs: number[]): Int32Array {
  const out = new Int32Array(xs.length);
  for (let i = 0; i < xs.length; i += 1) out[i] = xs[i] | 0;
  return out;
}

function toI64(xs: number[]): BigInt64Array {
  const out = new BigInt64Array(xs.length);
  for (let i = 0; i < xs.length; i += 1) out[i] = BigInt(xs[i] | 0);
  return out;
}

export async function loadTaskPresets(url: string): Promise<TaskPresetFile> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`failed to fetch task presets: ${r.status}`);
  return (await r.json()) as TaskPresetFile;
}

async function imageFileToCanvas(file: File): Promise<HTMLCanvasElement> {
  const bmp = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = bmp.width;
  canvas.height = bmp.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("failed to create canvas 2d context");
  ctx.drawImage(bmp, 0, 0);
  return canvas;
}

function letterboxToCanvas(src: HTMLCanvasElement, targetW: number, targetH: number): HTMLCanvasElement {
  const out = document.createElement("canvas");
  out.width = targetW;
  out.height = targetH;
  const ctx = out.getContext("2d");
  if (!ctx) throw new Error("failed to create letterbox context");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, targetW, targetH);
  const scale = Math.min(targetW / Math.max(1, src.width), targetH / Math.max(1, src.height));
  const nw = Math.max(1, Math.round(src.width * scale));
  const nh = Math.max(1, Math.round(src.height * scale));
  const x = Math.floor((targetW - nw) / 2);
  const y = Math.floor((targetH - nh) / 2);
  ctx.drawImage(src, 0, 0, src.width, src.height, x, y, nw, nh);
  return out;
}

function canvasToNormalizedCHW(canvas: HTMLCanvasElement): Float32Array {
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("failed to create image context");
  const { width, height } = canvas;
  const img = ctx.getImageData(0, 0, width, height);
  const rgba = img.data;
  const hw = width * height;
  const out = new Float32Array(3 * hw);
  for (let i = 0; i < hw; i += 1) {
    const r = rgba[i * 4] / 255.0;
    const g = rgba[i * 4 + 1] / 255.0;
    const b = rgba[i * 4 + 2] / 255.0;
    out[i] = (r - IMAGE_MEAN[0]) / IMAGE_STD[0];
    out[hw + i] = (g - IMAGE_MEAN[1]) / IMAGE_STD[1];
    out[2 * hw + i] = (b - IMAGE_MEAN[2]) / IMAGE_STD[2];
  }
  return out;
}

function patchifyImage(chw: Float32Array, width: number, height: number): { pixelValues: Float32Array; imageGridTHW: BigInt64Array } {
  const channel = 3;
  const gridH = Math.floor(height / PATCH_SIZE);
  const gridW = Math.floor(width / PATCH_SIZE);
  if (gridH % MERGE_SIZE !== 0 || gridW % MERGE_SIZE !== 0) {
    throw new Error(`grid not divisible by merge size: gridH=${gridH} gridW=${gridW} merge=${MERGE_SIZE}`);
  }
  const gridT = 1;
  const patchElems = channel * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE;
  const out = new Float32Array(gridT * gridH * gridW * patchElems);
  const hw = width * height;
  let tok = 0;
  for (let ghBlock = 0; ghBlock < gridH / MERGE_SIZE; ghBlock += 1) {
    for (let gwBlock = 0; gwBlock < gridW / MERGE_SIZE; gwBlock += 1) {
      for (let mh = 0; mh < MERGE_SIZE; mh += 1) {
        for (let mw = 0; mw < MERGE_SIZE; mw += 1) {
          const gh = ghBlock * MERGE_SIZE + mh;
          const gw = gwBlock * MERGE_SIZE + mw;
      const baseTok = tok * patchElems;
      let k = 0;
      for (let c = 0; c < channel; c += 1) {
        const cBase = c * hw;
        for (let t = 0; t < TEMPORAL_PATCH_SIZE; t += 1) {
          // For still images, GLM processor duplicates the same frame to fill temporal patches.
          for (let ph = 0; ph < PATCH_SIZE; ph += 1) {
            const y = gh * PATCH_SIZE + ph;
            const rowBase = cBase + y * width;
            for (let pw = 0; pw < PATCH_SIZE; pw += 1) {
              const x = gw * PATCH_SIZE + pw;
              out[baseTok + k] = chw[rowBase + x];
              k += 1;
            }
          }
        }
      }
      tok += 1;
        }
      }
    }
  }
  const imageGridTHW = toI64([gridT, gridH, gridW]);
  return { pixelValues: out, imageGridTHW };
}

function padToMaxSeq(preset: TaskPresetEntry, maxSeq: number): { inputIds: Int32Array; attentionMask: Int32Array; seqLen: number } {
  const inputIds = toI32(preset.input_ids);
  const attentionMask = toI32(preset.attention_mask);
  if (inputIds.length !== maxSeq || attentionMask.length !== maxSeq) {
    throw new Error(`task preset size mismatch: ids=${inputIds.length}, mask=${attentionMask.length}, max_seq=${maxSeq}`);
  }
  return { inputIds, attentionMask, seqLen: preset.seq_len | 0 };
}

export async function preprocessImageToModelInputs(
  imageFile: File,
  task: "document" | "text" | "table" | "formula",
  manifest: WebRuntimeManifest,
  presets: TaskPresetFile
): Promise<ModelInputs> {
  const preset = presets.tasks[task];
  if (!preset) throw new Error(`missing task preset: ${task}`);
  const { width, height } = manifest.image_size;
  const src = await imageFileToCanvas(imageFile);
  const boxed = letterboxToCanvas(src, width, height);
  const chw = canvasToNormalizedCHW(boxed);
  const { pixelValues, imageGridTHW } = patchifyImage(chw, width, height);
  const ids = padToMaxSeq(preset, manifest.max_seq_len);
  return {
    inputIds: ids.inputIds,
    attentionMask: ids.attentionMask,
    imageGridTHW,
    pixelValues,
    seqLen: ids.seqLen,
    tImg: presets.t_img,
    hiddenSize: presets.hidden_size,
  };
}

export { MERGE_SIZE, PATCH_SIZE, TEMPORAL_PATCH_SIZE };
