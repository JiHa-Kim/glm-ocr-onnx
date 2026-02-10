import type {
  GenerateOptions,
  GenerateResult,
  ModelInputs,
  WebRuntimeManifest,
} from "./types.js";

type SessionMap = Partial<
  Record<
    "vision" | "vision_quant" | "embed" | "rope" | "decode_prefill_kv" | "decode_step_kv",
    OrtSessionLike
  >
>;

const F16_TO_F32 = new Float32Array(65536);
let f16LutReady = false;

function ensureF16Lut(): void {
  if (f16LutReady) return;
  for (let i = 0; i < 65536; i += 1) {
    const s = (i >>> 15) & 0x1;
    const e = (i >>> 10) & 0x1f;
    const f = i & 0x3ff;
    let out: number;
    if (e === 0) {
      out = (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    } else if (e === 31) {
      out = f === 0 ? (s ? -Infinity : Infinity) : NaN;
    } else {
      out = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    F16_TO_F32[i] = out;
  }
  f16LutReady = true;
}

function f32ToF16Bits(v: number): number {
  if (!Number.isFinite(v)) return v < 0 ? 0xfc00 : 0x7c00;
  const s = v < 0 ? 0x8000 : 0;
  const av = Math.abs(v);
  if (av === 0) return s;
  if (av >= 65504) return s | 0x7bff;
  if (av < 6.103515625e-5) return s | Math.round(av / 5.960464477539063e-8);
  const exp = Math.floor(Math.log2(av));
  const mant = av / Math.pow(2, exp) - 1;
  const e = exp + 15;
  const m = Math.round(mant * 1024);
  return s | ((e & 0x1f) << 10) | (m & 0x3ff);
}

function f32ToF16Array(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i += 1) out[i] = f32ToF16Bits(src[i]);
  return out;
}

function dataAsFloat32(view: ArrayBufferView): Float32Array {
  if (view instanceof Float32Array) return view;
  if (view instanceof Uint16Array) {
    ensureF16Lut();
    const out = new Float32Array(view.length);
    for (let i = 0; i < view.length; i += 1) out[i] = F16_TO_F32[view[i]];
    return out;
  }
  if (view instanceof Int32Array) {
    const out = new Float32Array(view.length);
    for (let i = 0; i < view.length; i += 1) out[i] = view[i];
    return out;
  }
  return new Float32Array(view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength));
}

function argmax(v: ArrayBufferView): number {
  if (v instanceof Float32Array) {
    let bestIdx = 0;
    let best = v[0];
    for (let i = 1; i < v.length; i += 1) {
      const cur = v[i];
      if (cur > best) {
        best = cur;
        bestIdx = i;
      }
    }
    return bestIdx;
  }
  if (v instanceof Uint16Array) {
    ensureF16Lut();
    let bestIdx = 0;
    let best = F16_TO_F32[v[0]];
    for (let i = 1; i < v.length; i += 1) {
      const cur = F16_TO_F32[v[i]];
      if (cur > best) {
        best = cur;
        bestIdx = i;
      }
    }
    return bestIdx;
  }
  if (v instanceof Int32Array) {
    let bestIdx = 0;
    let best = v[0];
    for (let i = 1; i < v.length; i += 1) {
      const cur = v[i];
      if (cur > best) {
        best = cur;
        bestIdx = i;
      }
    }
    return bestIdx;
  }
  const f = dataAsFloat32(v);
  let bestIdx = 0;
  let best = f[0];
  for (let i = 1; i < f.length; i += 1) {
    const cur = f[i];
    if (cur > best) {
      best = cur;
      bestIdx = i;
    }
  }
  return bestIdx;
}

function isBigInt64Available(): boolean {
  return typeof BigInt64Array !== "undefined";
}

export class GlmOcrWebRuntime {
  manifest: WebRuntimeManifest;
  sessions: SessionMap = {};
  profile: string;
  ep: "webgpu" | "wasm";

  constructor(manifest: WebRuntimeManifest, profile = "full_browser_kv", ep: "webgpu" | "wasm" = "webgpu") {
    this.manifest = manifest;
    this.profile = profile;
    this.ep = ep;
  }

  static async fromUrl(url: string, profile = "full_browser_kv", ep: "webgpu" | "wasm" = "webgpu"): Promise<GlmOcrWebRuntime> {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`failed to fetch manifest: ${r.status}`);
    const m = (await r.json()) as WebRuntimeManifest;
    return new GlmOcrWebRuntime(m, profile, ep);
  }

  private graphUrl(graphKey: keyof WebRuntimeManifest["graphs"]): string {
    return `${this.manifest.base_url}${this.manifest.graphs[graphKey]}`;
  }

  private async createSession(graphKey: keyof WebRuntimeManifest["graphs"]): Promise<OrtSessionLike> {
    const opts = this.manifest.runtime[this.ep];
    return ort.InferenceSession.create(this.graphUrl(graphKey), {
      executionProviders: opts.executionProviders,
      logSeverityLevel: opts.logSeverityLevel,
    });
  }

  async init(): Promise<void> {
    const p = this.manifest.profiles[this.profile];
    if (!p) throw new Error(`unknown profile: ${this.profile}`);
    const needed = new Set(p.graphs);
    const graphKeys: Array<keyof WebRuntimeManifest["graphs"]> = [
      "vision",
      "vision_quant",
      "embed",
      "rope",
      "decode_prefill_kv",
      "decode_step_kv",
    ];
    const toLoad = graphKeys.filter((g) => needed.has(g));
    const loaded = await Promise.all(
      toLoad.map(async (g) => [g, await this.createSession(g)] as const)
    );
    for (const [g, s] of loaded) this.sessions[g] = s;
  }

  private async getSession(graphKey: keyof WebRuntimeManifest["graphs"]): Promise<OrtSessionLike> {
    const s = this.sessions[graphKey];
    if (s) return s;
    const created = await this.createSession(graphKey);
    this.sessions[graphKey] = created;
    return created;
  }

  private makeTensor(type: string, data: ArrayBufferView, dims: number[]): OrtTensorLike {
    return new ort.Tensor(type, data, dims);
  }

  private ensureI64(arr: BigInt64Array | Int32Array): BigInt64Array | Int32Array {
    if (arr instanceof BigInt64Array) return arr;
    if (!isBigInt64Available()) return arr;
    const out = new BigInt64Array(arr.length);
    for (let i = 0; i < arr.length; i += 1) out[i] = BigInt(arr[i]);
    return out;
  }

  private setI64At(arr: BigInt64Array | Int32Array, idx: number, value: number): void {
    if (arr instanceof BigInt64Array) {
      arr[idx] = BigInt(value);
      return;
    }
    arr[idx] = value;
  }

  private spliceEmbeds(
    tokenEmbeds: ArrayBufferView,
    imageEmbeds: ArrayBufferView,
    inputIds: Int32Array,
    imageTokenId: number,
    seqLen: number,
    tImg: number,
    hidden: number
  ): ArrayBufferView {
    let start = -1;
    for (let i = 0; i < seqLen; i += 1) {
      if (inputIds[i] === imageTokenId) {
        start = i;
        break;
      }
    }
    if (start < 0) throw new Error("image token block not found");
    const endExcl = start + tImg;
    if (endExcl > seqLen) throw new Error("invalid image token span");

    if (tokenEmbeds instanceof Uint16Array && imageEmbeds instanceof Uint16Array) {
      const out = tokenEmbeds.slice();
      out.set(imageEmbeds.subarray(0, tImg * hidden), start * hidden);
      return out;
    }
    const tokF = dataAsFloat32(tokenEmbeds);
    const imgF = dataAsFloat32(imageEmbeds);
    const outF = tokF.slice();
    outF.set(imgF.subarray(0, tImg * hidden), start * hidden);
    return this.manifest.dtype === "float16" ? f32ToF16Array(outF) : outF;
  }

  async generateGreedy(inputs: ModelInputs, opts: GenerateOptions): Promise<GenerateResult> {
    const sEmbed = await this.getSession("embed");
    const sRope = await this.getSession("rope");
    const sPref = await this.getSession("decode_prefill_kv");
    const sStep = await this.getSession("decode_step_kv");
    const imageTokenId = this.manifest.image_token_id;
    const eos = new Set(this.manifest.eos_token_ids || []);
    const nLayers = this.manifest.kv_cache.num_layers;
    const maxSeq = this.manifest.max_seq_len;
    const task = opts.task ?? "document";
    const visionPolicy = opts.visionPolicy ?? "fp16";
    const useEmbedTokenCache = opts.useEmbedTokenCache ?? true;

    const ids = inputs.inputIds.slice();
    const mask = inputs.attentionMask.slice();
    const i64Grid = this.ensureI64(inputs.imageGridTHW);
    const i64Mask = this.ensureI64(mask);
    const idsI64 = this.ensureI64(ids);
    const idType = idsI64 instanceof BigInt64Array ? "int64" : "int32";
    const i64Type = i64Mask instanceof BigInt64Array ? "int64" : "int32";
    const idTensor = this.makeTensor(idType, idsI64, [1, maxSeq]);
    const gridTensor = this.makeTensor(i64Grid instanceof BigInt64Array ? "int64" : "int32", i64Grid, [1, 3]);
    const maskTensor = this.makeTensor(i64Type, i64Mask, [1, maxSeq]);

    let imageEmbedsData: ArrayBufferView;
    if (inputs.imageEmbeds) {
      imageEmbedsData = this.manifest.dtype === "float16" ? f32ToF16Array(inputs.imageEmbeds) : inputs.imageEmbeds;
    } else {
      if (!inputs.pixelValues) throw new Error("pixelValues required for full-browser profile");
      const sVisionFp16 = await this.getSession("vision");
      let sVision = sVisionFp16;
      if (visionPolicy === "quant") {
        sVision = this.manifest.graphs.vision_quant ? await this.getSession("vision_quant") : sVisionFp16;
      } else if (visionPolicy === "table_quant") {
        if (task === "table" && this.manifest.graphs.vision_quant) {
          sVision = await this.getSession("vision_quant");
        } else {
          sVision = sVisionFp16;
        }
      } else if (visionPolicy === "auto" && task !== "text" && this.manifest.graphs.vision_quant) {
        sVision = await this.getSession("vision_quant");
      }
      const gridT = Number(
        inputs.imageGridTHW instanceof BigInt64Array ? inputs.imageGridTHW[0] : inputs.imageGridTHW[0]
      );
      const gridH = Number(
        inputs.imageGridTHW instanceof BigInt64Array ? inputs.imageGridTHW[1] : inputs.imageGridTHW[1]
      );
      const gridW = Number(
        inputs.imageGridTHW instanceof BigInt64Array ? inputs.imageGridTHW[2] : inputs.imageGridTHW[2]
      );
      const tokenCount = gridT * gridH * gridW;
      if (tokenCount <= 0) throw new Error(`invalid image grid: ${gridT},${gridH},${gridW}`);
      if (inputs.pixelValues.length % tokenCount !== 0) {
        throw new Error(`pixel_values size mismatch: len=${inputs.pixelValues.length}, token_count=${tokenCount}`);
      }
      const featDim = inputs.pixelValues.length / tokenCount;
      const pv = this.makeTensor(
        this.manifest.dtype === "float16" ? "float16" : "float32",
        this.manifest.dtype === "float16" ? f32ToF16Array(inputs.pixelValues) : inputs.pixelValues,
        [tokenCount, featDim]
      );
      const vo = await sVision.run({ pixel_values: pv, image_grid_thw: gridTensor }, ["image_embeds"]);
      imageEmbedsData = vo.image_embeds.data;
    }

    const embedOut = await sEmbed.run({ input_ids: idTensor }, ["token_embeds"]);

    const spliced = this.spliceEmbeds(
      embedOut.token_embeds.data,
      imageEmbedsData,
      ids,
      imageTokenId,
      inputs.seqLen,
      inputs.tImg,
      inputs.hiddenSize
    );

    const ropeOut = await sRope.run(
      {
        input_ids: idTensor,
        image_grid_thw: gridTensor,
        attention_mask: maskTensor,
      },
      ["position_ids", "rope_deltas"]
    );

    const pos = ropeOut.position_ids.data as BigInt64Array | Int32Array;
    const pastKeyNames = Array.from({ length: nLayers }, (_, i) => `past_key_${i}`);
    const pastValueNames = Array.from({ length: nLayers }, (_, i) => `past_value_${i}`);
    const presentKeyNames = Array.from({ length: nLayers }, (_, i) => `present_key_${i}`);
    const presentValueNames = Array.from({ length: nLayers }, (_, i) => `present_value_${i}`);
    const prefillOutNames = ["logits", ...pastKeyNames, ...pastValueNames];
    const pre = await sPref.run(
      {
        inputs_embeds: this.makeTensor(this.manifest.dtype === "float16" ? "float16" : "float32", spliced, [1, maxSeq, inputs.hiddenSize]),
        attention_mask: maskTensor,
        position_ids: this.makeTensor(pos instanceof BigInt64Array ? "int64" : "int32", pos, [3, 1, maxSeq]),
      },
      prefillOutNames
    );

    const past: ArrayBufferView[] = [];
    const pastDims: number[][] = [];
    for (let i = 0; i < nLayers; i += 1) {
      const keyName = pastKeyNames[i];
      const valueName = pastValueNames[i];
      past.push(pre[keyName].data);
      past.push(pre[valueName].data);
      pastDims.push(pre[keyName].dims);
      pastDims.push(pre[valueName].dims);
    }
    let nextLogits = pre.logits.data;

    const outIds: number[] = [];
    let writePos = inputs.seqLen;
    const stepOutNames = ["logits", ...presentKeyNames, ...presentValueNames];
    const stepPos = pos instanceof BigInt64Array ? new BigInt64Array(3) : new Int32Array(3);
    const cachePos = pos instanceof BigInt64Array ? new BigInt64Array(1) : new Int32Array(1);
    const oneId = pos instanceof BigInt64Array ? new BigInt64Array(maxSeq) : new Int32Array(maxSeq);
    const oneIdTensor = this.makeTensor(oneId instanceof BigInt64Array ? "int64" : "int32", oneId, [1, maxSeq]);
    const stepPosTensor = this.makeTensor(stepPos instanceof BigInt64Array ? "int64" : "int32", stepPos, [3, 1, 1]);
    const cachePosTensor = this.makeTensor(cachePos instanceof BigInt64Array ? "int64" : "int32", cachePos, [1]);
    const stepMaskTensor = this.makeTensor(i64Type, i64Mask, [1, maxSeq]);
    const embedTokenCache = new Map<number, ArrayBufferView>();
    const emptyCurEmb =
      this.manifest.dtype === "float16"
        ? new Uint16Array(inputs.hiddenSize)
        : new Float32Array(inputs.hiddenSize);
    const feeds: Record<string, OrtTensorLike> = {
      cur_emb: this.makeTensor(
        this.manifest.dtype === "float16" ? "float16" : "float32",
        emptyCurEmb,
        [1, 1, inputs.hiddenSize]
      ),
      attention_mask: stepMaskTensor,
      position_ids: stepPosTensor,
      cache_position: cachePosTensor,
    };

    while (outIds.length < opts.maxNewTokens && writePos < maxSeq) {
      const nextId = argmax(nextLogits);
      outIds.push(nextId);
      ids[writePos] = nextId;
      mask[writePos] = 1;
      this.setI64At(idsI64, writePos, nextId);
      this.setI64At(i64Mask, writePos, 1);
      if (eos.has(nextId)) break;

      // Build one-token embedding from full embedding table graph (static shape model).
      let curEmb: ArrayBufferView | undefined = undefined;
      if (useEmbedTokenCache) curEmb = embedTokenCache.get(nextId);
      if (!curEmb) {
        this.setI64At(oneId, 0, nextId);
        const oneEmb = await sEmbed.run({ input_ids: oneIdTensor }, ["token_embeds"]);
        const embData = oneEmb.token_embeds.data;
        if (embData instanceof Uint16Array) {
          curEmb = embData.slice(0, inputs.hiddenSize);
        } else {
          curEmb = dataAsFloat32(embData).slice(0, inputs.hiddenSize);
        }
        if (useEmbedTokenCache) embedTokenCache.set(nextId, curEmb);
      }

      if (stepPos instanceof BigInt64Array && pos instanceof BigInt64Array) {
        const p = pos[writePos];
        stepPos[0] = p;
        stepPos[1] = p;
        stepPos[2] = p;
        cachePos[0] = BigInt(writePos);
      } else {
        const p = Number((pos as Int32Array)[writePos]);
        (stepPos as Int32Array)[0] = p;
        (stepPos as Int32Array)[1] = p;
        (stepPos as Int32Array)[2] = p;
        (cachePos as Int32Array)[0] = writePos;
      }

      feeds.cur_emb = this.makeTensor(
        this.manifest.dtype === "float16" ? "float16" : "float32",
        curEmb,
        [1, 1, inputs.hiddenSize]
      );
      for (let i = 0; i < nLayers; i += 1) {
        feeds[pastKeyNames[i]] = this.makeTensor(this.manifest.dtype === "float16" ? "float16" : "float32", past[2 * i], pastDims[2 * i]);
        feeds[pastValueNames[i]] = this.makeTensor(this.manifest.dtype === "float16" ? "float16" : "float32", past[2 * i + 1], pastDims[2 * i + 1]);
      }
      const step = await sStep.run(feeds, stepOutNames);
      nextLogits = step.logits.data;
      for (let i = 0; i < nLayers; i += 1) {
        past[2 * i] = step[presentKeyNames[i]].data;
        past[2 * i + 1] = step[presentValueNames[i]].data;
      }
      writePos += 1;
    }

    return { tokenIds: outIds, endedByEos: outIds.length > 0 && eos.has(outIds[outIds.length - 1]) };
  }
}
