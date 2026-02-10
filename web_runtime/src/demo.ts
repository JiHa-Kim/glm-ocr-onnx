import { GlmOcrWebRuntime } from "./runtime.js";
import { loadTaskPresets, preprocessImageToModelInputs } from "./preprocess.js";
import type { ModelInputs } from "./types.js";

function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element: ${id}`);
  return el as T;
}

function toI32(xs: number[]): Int32Array {
  const out = new Int32Array(xs.length);
  for (let i = 0; i < xs.length; i += 1) out[i] = xs[i] | 0;
  return out;
}

function toF32(xs: number[]): Float32Array {
  const out = new Float32Array(xs.length);
  for (let i = 0; i < xs.length; i += 1) out[i] = xs[i];
  return out;
}

function toI64(xs: number[]): BigInt64Array {
  const out = new BigInt64Array(xs.length);
  for (let i = 0; i < xs.length; i += 1) out[i] = BigInt(xs[i] | 0);
  return out;
}

interface InputJson {
  input_ids: number[];
  attention_mask: number[];
  image_grid_thw: number[];
  pixel_values?: number[];
  image_embeds?: number[];
  seq_len: number;
  t_img: number;
  hidden_size: number;
  max_new_tokens?: number;
}

async function readJsonFile(file: File): Promise<InputJson> {
  const txt = await file.text();
  return JSON.parse(txt) as InputJson;
}

async function main(): Promise<void> {
  const status = byId<HTMLPreElement>("status");
  const out = byId<HTMLPreElement>("output");
  const runBtn = byId<HTMLButtonElement>("run");
  const inputFile = byId<HTMLInputElement>("input-json");
  const imageFile = byId<HTMLInputElement>("image-file");
  const profileEl = byId<HTMLSelectElement>("profile");
  const epEl = byId<HTMLSelectElement>("ep");
  const taskEl = byId<HTMLSelectElement>("task");
  const visionPolicyEl = byId<HTMLSelectElement>("vision-policy");

  runBtn.onclick = async () => {
    try {
      if (!inputFile.files || inputFile.files.length === 0) {
        throw new Error("select an input JSON file");
      }
      const profile = profileEl.value as "full_browser_kv" | "hybrid_server_vision_client_kv" | "hybrid_server_vision_client_prefill_only";
      const ep = epEl.value as "webgpu" | "wasm";
      const task = taskEl.value as "document" | "text" | "table" | "formula";
      const visionPolicy = visionPolicyEl.value as "auto" | "fp16" | "quant" | "table_quant";
      status.textContent = "loading manifest/runtime...";
      const runtime = await GlmOcrWebRuntime.fromUrl("./manifest.web.json", profile, ep);
      await runtime.init();

      let inputs: ModelInputs;
      let maxNewTokens = 256;
      if (imageFile.files && imageFile.files.length > 0) {
        status.textContent = "preprocessing image in browser...";
        const presets = await loadTaskPresets("./task_presets.json");
        inputs = await preprocessImageToModelInputs(imageFile.files[0], task, runtime.manifest, presets);
      } else {
        if (!inputFile.files || inputFile.files.length === 0) {
          throw new Error("select either an image file or an input JSON file");
        }
        status.textContent = "loading input json...";
        const j = await readJsonFile(inputFile.files[0]);
        inputs = {
          inputIds: toI32(j.input_ids),
          attentionMask: toI32(j.attention_mask),
          imageGridTHW: toI64(j.image_grid_thw),
          pixelValues: j.pixel_values ? toF32(j.pixel_values) : undefined,
          imageEmbeds: j.image_embeds ? toF32(j.image_embeds) : undefined,
          seqLen: j.seq_len | 0,
          tImg: j.t_img | 0,
          hiddenSize: j.hidden_size | 0,
        };
        maxNewTokens = j.max_new_tokens ?? 256;
      }

      status.textContent = "running greedy decode...";
      const res = await runtime.generateGreedy(inputs, {
        maxNewTokens,
        task,
        visionPolicy,
        useEmbedTokenCache: true,
      });
      out.textContent = JSON.stringify(res, null, 2);
      status.textContent = "done";
    } catch (e) {
      status.textContent = `error: ${(e as Error).message}`;
    }
  };
}

main().catch((e) => {
  byId<HTMLPreElement>("status").textContent = `fatal: ${(e as Error).message}`;
});
