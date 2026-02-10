export {};

declare global {
  interface OrtTensorLike {
    type: string;
    data: ArrayBufferView;
    dims: number[];
  }

  interface OrtSessionLike {
    run(
      feeds: Record<string, OrtTensorLike>,
      fetches?: string[]
    ): Promise<Record<string, OrtTensorLike>>;
  }

  interface OrtInferenceSessionStaticLike {
    create(
      modelUrl: string,
      options?: Record<string, unknown>
    ): Promise<OrtSessionLike>;
  }

  interface OrtLike {
    Tensor: new (
      type: string,
      data: ArrayBufferView,
      dims: readonly number[]
    ) => OrtTensorLike;
    InferenceSession: OrtInferenceSessionStaticLike;
    env?: Record<string, unknown>;
  }

  const ort: OrtLike;
}
