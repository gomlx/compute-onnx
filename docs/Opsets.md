# ONNX Opset Targeting & Upgrade Roadmap

## Current Target: Opset 21 (IR Version 9)

`compute-onnx` currently emits ONNX graphs targeting **`ai.onnx` Opset 21** (and `com.microsoft` version 1, IR version 9), configured in [`internal/graph/compile.go`](../internal/graph/compile.go).

Targeting Opset 21 provides broad compatibility across the ONNX Runtime versions and execution providers supported by `compute-onnx`:

- **Native CPU / CUDA**: works with both recent ONNX Runtime releases (e.g., 1.30/1.31, which support up to Opset 28) and older pinned releases (e.g., ORT 1.27 for CUDA ≤ 12.4).
- **ROCm / MIGraphX**: compatible with AMD's prebuilt ORT wheels (e.g., ORT 1.23.2).
- **ORT-Web (WASM / WebGPU / WebNN)**: compatible with browser builds where newer opset kernels often lag native providers.

Because ONNX Runtime rejects models whose `OpsetImport` version exceeds the runtime's maximum supported opset, bumping the target opset should only be done once all actively supported execution providers and runtime versions support it (or gated dynamically on the detected runtime version).

---

## Future Opset Updates

### Opset 29

When `compute-onnx` adopts **Opset 29** (once ONNX Runtime releases and target execution providers support it), the following decomposed operations in [`internal/graph/ops_unary.go`](../internal/graph/ops_unary.go) can be updated to emit native ONNX operators directly:

- **`Expm1`** ([ONNX spec](https://onnx.ai/onnx/operators/onnx__Expm1.html)):
  - Currently lowered to `Exp(x) - 1`.
  - Switching to native ONNX `Expm1` avoids catastrophic cancellation for small `|x|` near zero and reduces graph size.
- **`Log1p`** ([ONNX spec](https://onnx.ai/onnx/operators/onnx__Log1p.html)):
  - Currently lowered to `Log(x + 1)`.
  - Switching to native ONNX `Log1p` preserves precision for small `|x|` near zero and simplifies the lowering.
