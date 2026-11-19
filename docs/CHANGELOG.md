# Changelog
- 2026-11-18:
  - Implemented `Expm1` by lowering it to `Exp(x) - 1` (ONNX native `Expm1` requires Opset 29; backend targets Opset 21), and implemented `IsFinite` using ONNX `IsInf` and `IsNaN` (`!(IsInf(x) || IsNaN(x))`); registered both operations for ONNX Runtime backend capability checks (`internal/graph/ops.go`, `internal/graph/ops_unary.go`).
  - Documented ONNX Opset 21 targeting and future Opset 29 roadmap (`Expm1`, `Log1p`) in `docs/Opsets.md` and `README.md`, and updated `github.com/gomlx/compute` dependency.

- 2026-11-17:
  - Updated `RNGBitGenerator(state, shape, algorithm)` (`backend.go`, `internal/graph/ops_rng.go`, `ops_fused_test.go`) for `compute.RNGAlgorithm`: supports canonical Steele `RNGSplitMix64` (`DefaultRNGAlgorithm` on standard EPs), OpenXLA-exact `RNGThreeFry` (`ThreeFry2x32-20`), and pure 32-bit `[4]uint32` `ThreeFry2x32-20` as `RNGDefault` on `WebGPU` (avoiding `Uint64` GPU buffers); disabled `OpTypeRNGBitGenerator` capability on `MIGraphX`.
  - Renamed `Function.QuantizedEmbeddingLookup` stub to `QuantizedGather(x, indices, xQuantization)` (`internal/graph/ops_fused.go`, `docs/fused_ops.md`), updated `Slice`/`Gather` comments (`internal/graph/{ops_special,ops_gather}.go`), and bumped `github.com/gomlx/compute` dependency.

- 2026-11-16:
  - Added GitHub Actions CI workflow `.github/workflows/windows_amd64_tests.yaml` for `windows/amd64`.
  - Added Windows support in `internal/ort` (`ort_windows.go`, `cuda_copy_windows.go`, `hip_sync_windows.go`, `ORT_API_CALL` in `ort_wrapper.c`, build tags on `cuda_copy.go` and `hip_sync.go`, and cross-platform library path lookup in `ort_test.go`).

- 2026-11-15:
  - Added GitHub Actions CI workflows under `.github/workflows/` for `linux/amd64` (native CPU + WASM via `wasmbrowsertest`), `linux/arm64`, and `darwin/arm64` (`macos-latest`).
  - Cleared `AxisNames` on runtime output/buffer shapes (`internal/engine/{native,web}/{executable,buffer}.go`) so concrete `Buffer.Shape()` values match `shapes.Make(...)` without retaining symbolic dynamic axis names (fixing `TestCompliance/DynamicOps/DynamicRNGBitGenerator`).

- 2026-11-14:
  - Implemented `Function.ConstantFromBuffer` (`backend.go`, `internal/graph/{builder,function}.go`) reading direct host or device buffer contents directly into constant ONNX nodes and finalizing `buf` when `donate == true`.
  - Replaced `shapes.AnonymousAxis` (`"?"`), `Backend.UniqueAxisName`, and `Function.NameAnonymousDynamicAxes` with `shapes.UniqueAxisName`, `shapes.IsGeneratedAxisName`, and `shapes.AxisNamesCompatible` (`backend.go`, `internal/graph/{builder,compile,function,save,ops_*}.go`, `cmd/onnx_printer/main.go`). `#`-prefixed generated axes on ONNX graph inputs/outputs are encoded as unset `TensorShapeProto.Dimension` entries (round-tripped via `shapes.UniqueAxisName("#onnx")`), and `Executable.Execute` skips cross-parameter binding enforcement for `#`-prefixed axes so ORT validates concrete runtime compatibility.
  - Expanded operator and dynamic-shape support for full `hugot` / `onnx-gomlx` compatibility (PR #23):
    - Implemented `IsNaN` (`compute.OpTypeIsNaN` in `internal/graph/ops_unary.go` and `internal/graph/ops.go`) across `Float16`, `BFloat16`, `Float32`, and `Float64`.
    - Added `ConvGeneral` support for `inputDilations > 1` (e.g. transposed convolutions) and negative `paddings` via `Pad` / `DynamicPad`, as well as zero-sized static output shapes (`internal/graph/ops_convgeneral.go`).
    - Added `Gather` support for `sliceSizes[axis] > 1` and `sliceSizes[axis] == 0` on mapped axes via single-`GatherND` coordinate grid expansion, along with `shapeinference.Gather` validation, dynamic `indexVectorAxis == rank` reshaping, dynamic operand index clamping, and `shapes.AnonymousAxis` (`"?"`) support (`internal/graph/ops_gather.go`).
    - Added `Where` support for `dtypes.Bool` via `Uint8` casts around ONNX `Where` (preserving multidirectional broadcasting and anonymous dynamic axes), handled zero-sized target shapes in `BroadcastInDim` / `DynamicBroadcastInDim`, and fixed `DynamicReshape` when reshaping dynamic inputs to static target shapes (`internal/graph/ops_special.go`).
    - Cloned caller slices/arrays in `Function.Constant` (`internal/graph/function.go`, via `dtypes.CloneAnySlice`) so subsequent caller mutations do not corrupt deferred ONNX constant serialization, registered `compute.OpTypeParameter` and `compute.OpTypeConstant` in `internal/graph/ops.go`, and moved generic dynamic-shape tests into `compute/support/backendtest/dynamic.go`.
    - Added dynamic output shape support (and fixed zero-sized static output handling) in `RNGBitGenerator` (`internal/graph/ops_rng.go`), rounded float32-emulated integer `ConvGeneral` results before integer cast (`internal/graph/ops_convgeneral.go`), and returned `compute.ErrNotImplemented` in `FusedLayerNorm` (`internal/graph/ops_fused.go`) when `gamma == nil` over dynamic normalized dimensions.

- 2026-11-13:
  - Implemented `FusedTopK` (`compute.OpTypeFusedTopK` in `internal/graph/ops_fused.go`):
    - Lowers `FusedTopK` (`Axis`, `Smallest`, sorted output, lower-index tie-breaking) to ONNX `TopK` with `Int32` index output across static and dynamic shapes and all sortable numeric DTypes (including order-preserving mappings for `BFloat16`, `Uint16`, `Uint32`, and `Uint64`).
  - Implemented `NonZero` (`compute.OpTypeNonZero` in `internal/graph/ops_special.go` and `internal/graph/ops.go`):
    - Supports both `[count, rank]` (`Transpose: false`, default) and `[rank, count]` (`Transpose: true`) output layouts, 0D scalar inputs, and all numeric/boolean DTypes (with exact zero/non-zero conversions for `BFloat16`, `Int8`, `Int16`, `Uint16`, `Uint32`, `Uint64`, and `Float64`).
  - Added unique dynamic axis naming (`Backend.UniqueAxisName`, `Function.KnownDynamicAxisNames`, `Function.NameAnonymousDynamicAxes`) matching `gobackend`:
    - Mints unique `#nonzeros_<id>`, `#conv_<id>`, `#dot_<id>`, `#concat_<id>`, and `#pad_<id>` axis names for data-dependent and derived anonymous dynamic axes so downstream operations share consistent axis identity.
    - Tracks node/axis sources for known dynamic axis names (`Function.resolveDynamicAxisDim1D`), automatically emitting ONNX `Shape(start=axis, end=axis+1)` nodes when `BroadcastInDim`, `DynamicBroadcastInDim`, or `DynamicIota` reference known dynamic axis names without explicit `Value` nodes.
  - Fixed dynamic and anonymous (`?`) axis handling in `ConvGeneral`, `ReduceWindow`, `ArgMinMax`, `Pad`, `Slice`, `DynamicReshape`, `DynamicBroadcastInDim`, `DynamicIota`, `ShapeToONNX`, and `ONNXValueInfoToShape` (`internal/graph/`):
    - Preserved dynamic dimensions and axis names on intermediate NCHW shapes in `ConvGeneral` and `ReduceWindow` (as well as `ArgMinMax` and interior `Pad`), avoiding `shapes.Make` panics when `shapeinference.ConvGeneral` produces dynamic or `shapes.AnonymousAxis` spatial dimensions.
    - Encoded anonymous dynamic axes (`shapes.AnonymousAxis` / unnamed) as unset `TensorShapeProto.Dimension` entries in ONNX rather than synthetic `axis_<i>` `DimParam` symbols, and round-tripped both anonymous dynamic dimensions and zero-sized static dimensions in `ONNXValueInfoToShape`.
  - Fixed `FusedQuantizedDenseConfig.QuantizedMatmuls` -> `FusedQuantizedDenseConfig.QuantizedMatmul`

- 2026-11-12:
  - Updated `FusedQuantizedDense` (`internal/graph/ops_fused.go`) to the new `compute.FusedQuantizedDenseConfig` signature, adding full support for scalar, 1D per-channel, and 2D block (`BlockAxis` `0` and `1`, including non-divisible `BlockSize`) `QuantLinear` scales and zero-points across `DenseLayoutInputOutputs` and `DenseLayoutOutputsInput`.

- 2026-11-07:
  - Implemented full `Pad` and `DynamicPad` support (`internal/graph/ops_special.go`):
    - Added interior padding (`Interior > 0` and dynamic `InteriorValue`) by reshaping each interior-padded axis $D_i$ to $(D_i, 1)$, padding the inserted unit axis by $(0, I_i)$ with `fillValue`, reshaping back to $D_i \cdot (1 + I_i)$, and compensating the trailing $I_i$ elements in the end padding ($E_i - I_i$).
    - Added negative edge padding (cropping) support in `Pad` via `Slice` (and `BroadcastInDim` when all input elements along an axis are cropped out) combined with non-negative ONNX `Pad`.
  - Fixed `ShiftLeft`, `ShiftRightLogical`, and `Rem` for signed integer types, and implemented `ShiftRightArithmetic` (`internal/graph/ops_binary.go`, `internal/graph/ops.go`):
    - Cast signed integer inputs (`Int8`, `Int16`, `Int32`, `Int64`) to their unsigned counterparts (`Uint8`, `Uint16`, `Uint32`, `Uint64`) before ONNX `BitShift` and cast back to the signed output DType.
    - Implemented `ShiftRightArithmetic` using logical right shift for unsigned types and two's-complement bitwise inversion (`x < 0 ? ~((~x) >> s) : (x >> s)`) for signed integer types.
    - Enabled `fmod = 1` on ONNX `Mod` for all DTypes so `Rem` on negative signed integers preserves dividend sign (C/Go/XLA semantics).
