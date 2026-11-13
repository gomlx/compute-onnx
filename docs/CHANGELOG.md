# Changelog
- 2026-11-13:
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
