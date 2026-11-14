# AGENTS.md — project documentation for AI coding agents

## Project Overview

`github.com/gomlx/compute-onnx` provides an **ONNX Runtime (ORT)** compute backend implementation for [GoMLX](https://github.com/gomlx/gomlx).

### Relationship to `gomlx/compute` and `gomlx/gomlx`

- **[`github.com/gomlx/gomlx`](https://github.com/gomlx/gomlx)**: High-level ML framework in Go (tensors, layers, automatic differentiation, optimizers, model training loops). GoMLX delegates lower-level graph compilation and execution to a `compute.Backend`.
- **[`github.com/gomlx/compute`](https://github.com/gomlx/compute)**: The foundational execution and backend abstraction interface (`compute.Backend`, `compute.Builder`, `compute.Function`, `compute.Executable`, `compute.Buffer`). It defines the unified IR graph and operator set that execution backends implement.
- **`github.com/gomlx/compute-onnx` (This Repository)**: Implements `compute.Backend` by translating GoMLX computation graphs into ONNX Model Protocol Buffers (`onnx.ModelProto`) at runtime and executing them using ONNX Runtime.

When navigating the codebase of any of the related projects, read their .agents/AGENTS.md file to get more information about
the structure of the codebase and their specific conventions.

### Documentation

- Changes should always be recored in the file `docs/CHANGELOG.md`.
- Please update `.agents/AGENTS.md` with any information relevant to agents (including yourself) to make their life easier. The file is usually linked to `GEMINI.md` or `CLAUDE.md`.

### Compliance Tests & Benchmarks

- If tests are needed, if they are generic, and can be used by other backends, prefer to place them along the "compliance tests" in `github.com/gomlx/compute/support/backendtest`.
- Run tests in the following fashion:
  1. `GOMLX_BACKEND=onnx:cpu go test ./...`, to force the CPU execution path.
  2. `GOMLX_BACKEND=onnx:cuda go test ./...`, if a GPU is available.
  3. `GOOS=js GOARCH=wasm GOMLX_BACKEND=onnx:wasm go test -exec wasmbrowsertest ./...`, to test wasm (see details in `docs/ort-web.md`)

### Follow Existing Patterns

Before writing new code, read neighboring files in the same package to understand the established
patterns (buffer management, dtype dispatch, parallelization, etc.). Reuse existing infrastructure
rather than writing ad-hoc implementations. When in doubt, match the style and approach of the
closest existing operation.

### Avoid Large Files (Split Per-Op Implementations > 200 Lines)

Keep source and assembly files focused and easy to browse so it is immediately clear from directory
listings which operations are implemented:
- **Avoid monolithic files**: Do not keep appending new operations into a single large file.
- **Per-op files for implementations > 200 lines**: Whenever an operation (or closely related op family) requires more than ~200 lines of Go, place it in its own dedicated file.

### Copyright Notes

Normal code files are prefixed with the following copyright line:

```
// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0
```

Auto-generated files don't need a copyright, but should include a comment with the tool use to generate them.


---

## Package Directory Structure

```text
github.com/gomlx/compute-onnx/
├── backend.go              # Public compute.Backend implementation facade & platform registration
├── save.go                 # Public SaveModel / LoadModel export and import helpers
├── doc.go                  # Package-level documentation
│
├── internal/
│   ├── graph/              # Pure Go AST graph construction & ONNX protobuf compilation
│   │   ├── node.go         # Node IR definitions
│   │   ├── builder.go      # compute.Builder implementation
│   │   ├── function.go     # compute.Function implementation
│   │   ├── compile.go      # AST -> onnx.ModelProto translator and shape inference
│   │   ├── save.go         # Protobuf serialization, tensor renaming, and shape parsing
│   │   ├── schedulingbarrier.go # Scheduling barrier implementation (no-op dependency injection)
│   │   ├── ops.go          # Operator registry & generic binary/unary/comparison helpers
│   │   └── ops_*.go        # GoMLX -> ONNX operator lowerings (DotGeneral, ConvGeneral, Scatter, etc.)
│   │
│   ├── engine/             # Execution engines (Runtime compilation, session management, buffers)
│   │   ├── native/         # Native desktop/server execution engine (CGO + ONNX Runtime C API)
│   │   │   ├── session.go  # DynamicAdvancedSession initialization, options, logging, and error wrapping
│   │   │   ├── executable.go # Native compute.Executable implementation (CPU & CUDA execution paths)
│   │   │   └── buffer.go   # compute.Buffer implementation (CPU host tensors & GPU OrtValue wrappers)
│   │   └── web/            # [Planned] ORT Web execution engine (WASM / JS via syscall/js)
│   │
│   ├── device/             # Platform & hardware detection
│   │   ├── cuda/           # CUDA Toolkit, cuDNN, and NVIDIA GPU auto-detection
│   │   │   ├── detect_linux.go   # Linux /dev/nvidia*, nvidia-smi, ldconfig, and search paths
│   │   │   ├── detect_other.go   # Non-Linux stubs
│   │   │   └── detect_linux_test.go
│   │   └── web/            # [Planned] WebGPU / WebGL browser environment detection
│   │
│   ├── ort/                # Low-level CGO wrapper for ONNX Runtime C API (onnxruntime_c_api.h)
│   ├── pool/               # Memory arena & slice pooling for CPU tensor re-use
│   └── cmd/                # Maintenance tooling
│       ├── update_c_ort/   # Code generator updating ONNX Runtime C headers
│       └── update_protos/  # Code generator fetching official ONNX .proto definitions
│
├── support/
│   ├── onnxruntime/        # Prebuilt libonnxruntime downloader & installer helper
│   └── protos/             # Auto-generated Go Protocol Buffer bindings for ONNX (onnx-ml.pb.go)
│
└── cmd/
    ├── onnx_printer/       # Diagnostic CLI tool for printing ONNX model graph structure
    └── onnxruntime_installer/ # CLI installer for downloading official ORT binaries
```

---

## Architectural Separation & Execution Lifecycle

### 1. Graph Construction & Compilation (`internal/graph`)
- **Pure Go**: The `internal/graph` package has **no CGO dependencies**. It builds the DAG of `*Node` operations and translates them directly into an `onnx.ModelProto` (IR version 9, opset 21).
- **Decoupled Compiler**: `graph.Builder` accepts a `CompilerFn` callback. The graph package does not know whether the serialized protobuf will be executed natively via CGO (`internal/engine/native`) or inside a browser via WebAssembly (`internal/engine/web`).
- **Graph Optimization**: Constant folding, shape propagation, and sub-graph fusing (e.g. `ConvGeneral`, `DotGeneral`, `SelectAndScatterMax`) happen during the AST lowering phase before protobuf marshaling.

### 2. Native CGO Execution Engine (`internal/engine/native`)
- **Session Management (`session.go`)**: Manages `ort.DynamicAdvancedSession` instances. Configures execution providers (`CUDAExecutionProvider` with default stream copying, thread pool settings, log severities).
- **Execution Paths (`executable.go`)**:
  - **CPU Path**: Uses pre-allocated input/output value slices and thread-safe recycling (`reusableWrappers`) to avoid allocations across execution steps.
  - **CUDA Path**: Uses `IoBinding` to bind device pointers directly to graph inputs and outputs, bypassing host memory transfers.

### 3. Device & Hardware Detection (`internal/device/cuda`)
- Dynamically checks `/dev/nvidia*`, `nvidia-smi`, `ldconfig -p`, and search paths (`LD_LIBRARY_PATH`, `CUDA_PATH`, `CONDA_PREFIX`, standard `/usr/local/cuda-*` targets) for `libcudart.so` and `libcudnn.so`.
- Isolates platform-specific build tags (`//go:build linux`, `//go:build !linux`) so that the root package and higher-level code remain clean and portable.

---

## Nuanced Performance Details & Transfer Costs

### 1. GPU (CUDA) Zero-Copy Execution vs Host PCIe Transfers
- **In-Memory GPU Retention**: When executing on CUDA, `Executable.executeCUDA` produces `native.Buffer` objects backed by `GpuTensorWrapper` containing GPU-resident `OrtValue` handles.
- **Zero-Copy Pipeline**: If the output of one computation graph is passed as input to the next graph on GPU, `IoBinding.BindInput` binds the existing device memory pointer directly without any Host-to-Device (H2D) or Device-to-Host (D2H) PCIe memory copy.
- **Transfer Cost Nuance**:
  - Calling `buffer.Data()` on a GPU buffer returns an error because direct pointer access across PCIe without explicit staging is disallowed.
  - Calling `buffer.ToFlatData(slice)` performs an explicit synchronous `cudaMemcpy` (D2H) via `ort.CopyGPUToHost`. This incurs PCIe bandwidth latency and pipeline synchronization stalls. Keep data on-device as long as possible during training/inference loops.

### 2. CPU Slice Pooling & Memory Recycling
- **Buffer Reuse**: On the CPU path, `native.Buffer` instances are backed by host memory. When a buffer is finalized via `buffer.Finalize()`, its underlying `OrtTensorWrapper` is returned to the executable's `reusableWrappers` pool.
- **Zero Allocation Loops**: Subsequent `Execute()` calls on static-shaped graphs reuse pooled output wrappers directly, resulting in zero heap allocations per step.
- **Dynamic Shapes Exception**: If graph outputs have dynamic dimensions (e.g. batch size `shapes.DynamicDim`), output shapes cannot be statically predicted before execution. ORT dynamically allocates the output tensor during `session.Run()`, which is then wrapped in a new `Buffer` and destroyed upon finalization rather than pooled.

### 3. Precision Conversions (`Float16` & `BFloat16`)
- **CPU Path**: ONNX Runtime C API does not have direct native Go primitive types for 16-bit floats. On CPU, `float16` and `bfloat16` tensors are widened to `float32` during tensor construction (`float16ToFloat32` / `bfloat16ToFloat32`) and converted back on extraction.
- **GPU Path**: `ort.CopyHostToGPU` and `ort.CopyGPUToHost` convert and copy 16-bit floats through float32 staging buffers when interfacing with Go host slices, but internal GPU kernel computation runs at full device float16 performance.

### 4. Concurrency & Thread Locking
- **OS Thread Locking**: `session.Run()` calls `runtime.LockOSThread()` / `UnlockOSThread()` to prevent the Go runtime scheduler from migrating goroutines between OS threads during low-level CGO runtime execution.
- **Wrapper Recycling Mutex**: Executable wrapper recycling and buffer finalization are guarded by `mu sync.Mutex`, allowing safe concurrent finalization from Go garbage collection finalizers.

### 5. Dynamic Shapes & Data-Dependent Axis Naming
- **Unique Axis Names**: `Backend.UniqueAxisName` and `Function.NameAnonymousDynamicAxes` mint unique `#<base>_<id>` axis names (e.g., `#nonzeros_1`, `#conv_2`, `#dot_3`, `#concat_4`, `#pad_5`) for data-dependent and derived anonymous dynamic axes, matching `gobackend` so downstream operations see matching axis identities.
- **Name-Only Dynamic Dimension Resolution**: `Function` records the first `(node, axis)` source of each named dynamic axis (`dynamicAxisSources`), allowing `BroadcastInDim`, `DynamicBroadcastInDim`, `DynamicReshape`, and `DynamicIota` to resolve known dynamic axis names via ONNX `Shape(start=axis, end=axis+1)` when no explicit dimension `Value` is provided.
- **Anonymous Axes in ONNX**: Explicit `shapes.AnonymousAxis` (`"?"`) dimensions on parameters/IO are encoded as unset `TensorShapeProto.Dimension` entries in ONNX rather than named `DimParam` symbols so independent anonymous axes are never unified by ONNX shape inference. Avoid decomposing an op on an input with `"?"` into multi-operand GoMLX binary ops on two `"?"`-shaped intermediates (because `shapes.UnifyAxisName("?", "?")` rightly errors); prefer unary/cast wrappers or static broadcast operands.

### 6. Notable Operator Lowering Nuances (`internal/graph`)
- **`Constant` Defensive Cloning**: `Function.Constant` clones caller-provided slices/arrays on cache miss (`dtypes.CloneAnySlice`), because upstream callers (e.g. `onnx-gomlx`) may mutate or reuse backing buffers before `CompileToProto` serializes `Node.flatValue`.
- **`ConvGeneral` Input Dilations & Cropping**: `inputDilations > 1` (e.g. transposed convolutions) and negative `paddings` are lowered via `Pad` (static spatial axes) or `DynamicPad` (dynamic spatial axes) before delegating to ONNX `Conv`.
- **`Gather` Window Expansion**: Non-collapsed mapped axes (`sliceSizes[axis] > 1`) are lowered into a single `GatherND` by adding a static coordinate offset grid `[1..1, W_0..W_{M-1}, V]` to reshaped clamped `startIndices`, avoiding $O(\prod W_m)$ recursive `Gather` + `Concatenate` subgraphs and supporting `shapes.AnonymousAxis` (`"?"`).
- **Zero-Sized `BroadcastInDim` & `Bool` `Where`**: Static empty `BroadcastInDim` (and `DynamicBroadcastInDim` with static `0` dimensions) bypass ONNX `Expand` (whose constant folding treats a `0` target dim as `1`). `Where` on `dtypes.Bool` casts `onTrue`/`onFalse` through `dtypes.Uint8` around a single ONNX `Where` because ORT lacks a native `Bool` `Where` kernel.
