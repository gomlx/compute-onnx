# AGENTS.md — project documentation for AI coding agents

## Project Overview

`github.com/gomlx/compute-onnx` provides an **ONNX Runtime (ORT)** compute backend implementation for [GoMLX](https://github.com/gomlx/gomlx).

### Relationship to `gomlx/compute` and `gomlx/gomlx`

- **[`github.com/gomlx/gomlx`](https://github.com/gomlx/gomlx)**: High-level ML framework in Go (tensors, layers, automatic differentiation, optimizers, model training loops). GoMLX delegates lower-level graph compilation and execution to a `compute.Backend`.
- **[`github.com/gomlx/compute`](https://github.com/gomlx/compute)**: The foundational execution and backend abstraction interface (`compute.Backend`, `compute.Builder`, `compute.Function`, `compute.Executable`, `compute.Buffer`). It defines the unified IR graph and operator set that execution backends implement.
- **`github.com/gomlx/compute-onnx` (This Repository)**: Implements `compute.Backend` by translating GoMLX computation graphs into ONNX Model Protocol Buffers (`onnx.ModelProto`) at runtime and executing them using ONNX Runtime.

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
