# Fused Operations in ONNX Runtime Backend (`compute-onnx`)

This document describes the implementation, mapping, and limitations of fused operations (`compute.FusedOps`) in `github.com/gomlx/compute-onnx`, as well as potential ONNX fused operations for future backend additions.

---

## 1. Overview of `compute.Backend` Fused Ops Mapping

GoMLX defines a set of high-level optional fused operations in `compute.FusedOps` (`github.com/gomlx/compute/fused_ops.go`). Backends indicate their support for these operations in `Capabilities().Operations`. If supported, GoMLX emits the backend's fused operation directly; if unsupported, GoMLX automatically decomposes the fused operation into standard primitive graph nodes.

Below is the summary mapping of `compute.Backend` fused operations to ONNX operators:

| `compute.Backend` Fused Op | ONNX Operator / Graph Mapping | Support Status in `compute-onnx` |
| :--- | :--- | :--- |
| `FusedSoftmax` | Standard ONNX `Softmax` (opset 13+) | **Supported** |
| `FusedGelu` | Standard ONNX `Gelu` (opset 20+) | **Supported** |
| `FusedLayerNorm` | Standard ONNX `LayerNormalization` (opset 17+) | **Supported** (with trailing contiguous axes restriction) |
| `FusedDense` | ONNX `MatMul` / `Einsum` + `Add` + Activation (`Relu`, `Gelu`, `Sigmoid`/`Mul`, `HardSwish`, `Tanh`) | **Supported** |
| `FusedScaledDotProductAttention` | ONNX Graph / `Softmax( (Q @ K^T) * scale + mask + bias ) @ V` (Fused by ORT Session Optimizer) | **Supported** (Forward Pass) |
| `FusedScaledDotProductAttentionVJP` | Direct backward gradient graph | **Not Implemented** (Falls back to decomposed VJP) |
| `FusedAttentionQKVProjection` | ONNX `MatMul` + `Slice` (Q, K, V) + `Add` (bias) | **Supported** |
| `FusedQuantizedDense` | Dequantization (`ConvertDType`, `Sub`, `Mul`) + `FusedDense` | **Supported** (for `QuantLinear`) |
| `QuantizedEmbeddingLookup` | Row gather with on-the-fly block dequantization | **Not Implemented** (GGML format unsupported in standard ONNX) |

---

## 2. Detailed Operator Mapping & Conversion Limitations

### 2.1 `FusedSoftmax`
- **GoMLX Interface**: `FusedSoftmax(x Value, axis int) (Value, error)`
- **ONNX Operator**: Standard ONNX `Softmax` (opset 13+)
- **Attributes**: `axis` (int64)
- **Shortcomings & Limitations**:
  - `axis` must be non-negative (`0 <= axis < rank`). Negative axis indices must be normalized by the caller before invoking `FusedSoftmax`.

---

### 2.2 `FusedGelu`
- **GoMLX Interface**: `FusedGelu(x Value, exact bool) (Value, error)`
- **ONNX Operator**: Standard ONNX `Gelu` (opset 20+)
- **Attributes**: `approximate` string (`"none"` for exact GELU using `erf`, `"tanh"` for tanh approximation).
- **Shortcomings & Limitations**:
  - Requires floating-point dtypes (`Float32`, `Float64`, `Float16`, `BFloat16`).

---

### 2.3 `FusedLayerNorm`
- **GoMLX Interface**: `FusedLayerNorm(x Value, axes []int, epsilon float64, gamma, beta Value) (Value, error)`
- **ONNX Operator**: Standard ONNX `LayerNormalization` (opset 17+)
- **Attributes**: `axis` (int64), `epsilon` (float32)
- **Inputs**: `X`, `Scale` (gamma), optional `Bias` (beta). If `gamma` is `nil`, a constant tensor of 1s matching the normalized shape is generated.
- **Shortcomings & Limitations**:
  - **Trailing Contiguous Axes Restriction**: ONNX `LayerNormalization` standard operator normalizes across all axes starting from `axis` through `rank - 1`. Therefore, GoMLX `axes` MUST form a contiguous trailing suffix `[rank-k, ..., rank-1]`.
  - If `axes` does not form a contiguous trailing suffix (e.g., non-contiguous axes like `[0, 2]` for a 3D tensor or non-suffix axes), `FusedLayerNorm` returns `compute.ErrNotImplemented`, triggering GoMLX's decomposed fallback path.

---

### 2.4 `FusedDense`
- **GoMLX Interface**: `FusedDense(x, weight, bias Value, options DenseConfig) (Value, error)`
- **ONNX Operator Mapping**: Expressed via ONNX `DotGeneral` (`MatMul`/`Einsum`), optional ONNX `Add` for bias, and activation operator (`Relu`, `Gelu`, `Mul(x, Sigmoid(x))` for `Silu`, `HardSwish`, `Tanh`).
- **ONNX Runtime Fusion**: ONNX Runtime's internal session graph optimizer (`GemmFusion`, `MatMulAddFusion`, `ActivationFusion`) automatically merges these sequential ONNX nodes into optimized fused execution kernels at runtime.
- **Supported Activations**: `ActivationNone`, `ActivationRelu`, `ActivationGelu`, `ActivationSilu`, `ActivationHardSwish`, `ActivationTanh`.
- **Supported Weight Layouts**: `DenseLayoutInputOutputs` (`[in_features, out_features...]`) and `DenseLayoutOutputsInput` (`[out_features..., in_features]`).

---

### 2.5 `FusedScaledDotProductAttention`
- **GoMLX Interface**:
  ```go
  FusedScaledDotProductAttention(
      query, key, value Value,
      axesLayout AttentionAxesLayout,
      options *ScaledDotProductAttentionConfig) (output Value, statesForVJP []Value, err error)
  ```
- **ONNX Operator Mapping**: Expressed as an optimized computational sub-graph: `Softmax( (query @ key^T) * scale + mask + bias ) @ value`.
- **Relationship to ONNX's `com.microsoft.MultiHeadAttention` and `com.microsoft.PackedAttention`**:
  - **Why not target `com.microsoft.MultiHeadAttention` or `com.microsoft.PackedAttention` directly for general SDPA?**
    1. **Input Expectations**: `com.microsoft.PackedAttention` expects a single 3D tensor of shape `[batch_size, sequence_length, query_hidden + 2 * kv_hidden]` combining packed Q, K, and V projections before head splitting. `com.microsoft.MultiHeadAttention` expects full input projections and weight matrices. `FusedScaledDotProductAttention` in GoMLX receives separate, pre-projected 4D Query, Key, and Value tensors (`[B, H, S, D]` or `[B, S, H, D]`).
    2. **ONNX Runtime Dynamic Graph Fusion**: When `FusedScaledDotProductAttention` is constructed using standard ONNX graph operations (`MatMul`, `Mul`, `Add`/`Where`, `Softmax`), ONNX Runtime's session optimizer (`AttentionFusion` transformer) automatically detects the attention sub-graph at runtime and fuses it into native execution provider kernels (such as cuDNN FlashAttention on GPU or SIMD MHA on CPU).
    3. **Portability**: Generating standard ONNX nodes preserves 100% numerical portability across all ONNX Runtime execution providers (CPU, CUDA, DirectML, OpenVINO) without relying on provider-specific contrib operator schema variations.
- **Layouts Supported**: Both `AttentionAxesLayoutBHSD` (`[B, H, S, D]`) and `AttentionAxesLayoutBSHD` (`[B, S, H, D]`).
- **Features Implemented**:
  - **Grouped Query Attention (GQA)**: Automatically expands Key and Value heads when `numHeadsKV < numHeadsQ`.
  - **Custom Scale**: Custom scale factor or default \(1 / \sqrt{\text{headDim}}\).
  - **Causal Masking**: Adds lower-triangular causal mask with large negative score offsets (`-1e9`).
  - **Boolean / Float Attention Mask**: Full support for boolean masks (using ONNX `Where`) and additive score masks.
  - **Attention Bias**: ALiBi / relative-position additive score bias.
- **Shortcomings & Limitations**:
  - **Per-Batch Sequence Length Tensors (`QuerySeqLen` / `KeyValueSeqLen`)**: Explicit per-batch sequence length tensors are not supported in graph-level SDPA and will return `ErrNotImplemented`.
  - **Fused VJP (`FusedScaledDotProductAttentionVJP`)**: Fused backward pass returns `ErrNotImplemented`. GoMLX automatically uses decomposed attention for gradient calculation during training.

---

### 2.6 `FusedAttentionQKVProjection`
- **GoMLX Interface**:
  ```go
  FusedAttentionQKVProjection(
      x, wQKV, biasQ, biasK, biasV Value,
      queryDim, keyValueDim int) (query, key, value Value, err error)
  ```
- **ONNX Operator Mapping**: `DotGeneral` (`x @ wQKV`), ONNX `Slice` to split into Query, Key, and Value tensors, and optional ONNX `Add` for per-projection biases.

---

### 2.7 `FusedQuantizedDense`
- **GoMLX Interface**:
  ```go
  FusedQuantizedDense(x, weights, bias Value, weightsQuantization *Quantization, activation ActivationType) (Value, error)
  ```
- **ONNX Operator Mapping**: Dequantizes integer weights to float32 using `floatWeight = (weights - zeroPoint) * scale`, then delegates to `FusedDense`.
- **Shortcomings & Limitations**:
  - Currently supports `QuantLinear` (affine quantization). `QuantNF4` and `QuantGGML` block formats return `ErrNotImplemented`.

---

### 2.8 `QuantizedEmbeddingLookup`
- **Shortcomings & Limitations**:
  - GGML block quantization formats (`Q4_0`, `Q8_0`, `K-quants`) are native to `llama.cpp` / `ggml` and are not part of standard ONNX specifications. Returns `ErrNotImplemented` to trigger GoMLX's decomposed dequantization path.

---

## 3. Other ONNX Fused Operations for Future Consideration

The ONNX standard specification (and the `com.microsoft` ONNX Runtime execution provider extension domain) includes additional fused operators that could be considered for future GoMLX `compute.Backend` interface extensions:

### 3.1 Normalization Operators
- **`GroupNormalization`** (Standard ONNX opset 18+):
  - Normalizes channels within groups (used heavily in diffusion models like Stable Diffusion).
  - Standard ONNX op: `GroupNormalization(X, num_groups, scale, bias)`.
- **`RMSNormalization` / `RmsNorm`** (`com.microsoft` domain):
  - Root Mean Square Layer Normalization used in modern LLMs (LLaMA, Mistral, Gemma).
  - Formula: \(y = \frac{x}{\sqrt{\text{Mean}(x^2) + \epsilon}} \cdot \gamma\).
- **`SkipLayerNormalization`** (`com.microsoft` domain):
  - Fuses elementwise `Add(X, skip)` + `LayerNormalization` (standard transformer residual connection block).

### 3.2 Activation & Math Fused Operators
- **`FastGelu` / `BiasGelu`** (`com.microsoft` domain):
  - Fuses bias addition + GELU activation into a single kernel (`BiasGelu(X, bias)`).
- **`BiasGeluGrad` / `GeluGrad`** (`com.microsoft` domain):
  - Backward pass fused gradient kernels for GELU.
- **`QuickGelu`** (`com.microsoft` domain):
  - Fast approximation of GELU using Sigmoid: \(x \cdot \text{Sigmoid}(1.702 \cdot x)\) (used in CLIP models).
- **`BiasRelu` / `BiasSilu`** (`com.microsoft` domain):
  - Fuses bias addition with ReLU or SiLU activations.

### 3.3 Matrix Multiplication & Convolution Fused Operators
- **`FusedMatMul` / `MatMulAdd`** (`com.microsoft` domain):
  - Fuses matrix multiplication with transpose flags, alpha/beta scaling, and additive bias.
- **`QLinearMatMul` / `QLinearConv`** (Standard ONNX opset 10+):
  - Quantized matrix multiplication and convolution operating directly on int8/uint8 inputs and outputs without explicit dequantization to float.
- **`MatMulNBits`** (`com.microsoft` domain):
  - Native 4-bit / 2-bit weight-only quantized matrix multiplication kernel optimized for LLM inference on CPU and CUDA.
- **`ConvAdd` / `FusedConv`** (`com.microsoft` domain):
  - Fuses Convolution + Bias Addition + Activation (`Relu`/`Elementwise`).

### 3.4 Attention & Transformer Blocks
- **`MultiHeadAttention` / `PackedAttention`** (`com.microsoft` domain):
  - Fuses Q/K/V projections, relative position embeddings, causal/padding masking, and scaled dot product attention into a single CUDNN / FlashAttention kernel.
- **`RotaryEmbedding` (RoPE)** (`com.microsoft` domain):
  - Fuses rotary position embedding calculations (\(\cos\) and \(\sin\) frequency rotations) directly into Q and K tensors before attention.
