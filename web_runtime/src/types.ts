export interface WebRuntimeManifest {
  model_id: string;
  dtype: "float16" | "float32";
  max_seq_len: number;
  image_size: { width: number; height: number };
  image_token_id: number;
  eos_token_ids: number[];
  kv_cache: {
    num_layers: number;
    num_key_value_heads: number;
    head_dim: number;
    max_cache_len: number;
  };
  base_url: string;
  graphs: {
    vision: string;
    vision_quant?: string;
    embed: string;
    rope: string;
    decode_prefill_kv: string;
    decode_step_kv: string;
  };
  runtime: {
    webgpu: { executionProviders: string[]; logSeverityLevel: number };
    wasm: { executionProviders: string[]; logSeverityLevel: number };
  };
  profiles: Record<
    string,
    {
      description: string;
      graphs: string[];
      requires_image_embeds_input: boolean;
      estimated_weight_gb: number;
    }
  >;
}

export interface ModelInputs {
  inputIds: Int32Array;
  attentionMask: Int32Array;
  imageGridTHW: BigInt64Array | Int32Array;
  pixelValues?: Float32Array;
  imageEmbeds?: Float32Array;
  seqLen: number;
  tImg: number;
  hiddenSize: number;
}

export interface GenerateOptions {
  maxNewTokens: number;
  task?: "document" | "text" | "table" | "formula";
  visionPolicy?: "auto" | "fp16" | "quant" | "table_quant";
  useEmbedTokenCache?: boolean;
}

export interface GenerateResult {
  tokenIds: number[];
  endedByEos: boolean;
}

export interface TaskPresetEntry {
  task: "document" | "text" | "table" | "formula";
  prompt: string;
  input_ids: number[];
  attention_mask: number[];
  image_grid_thw: number[];
  seq_len: number;
}

export interface TaskPresetFile {
  model_id: string;
  max_seq_len: number;
  t_img: number;
  hidden_size: number;
  image_size: { width: number; height: number };
  tasks: Record<string, TaskPresetEntry>;
}
