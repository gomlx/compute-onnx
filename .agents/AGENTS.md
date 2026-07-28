# AGENTS.md — project documentation for AI coding agents

## Project Overview

`github.com/gomlx/compute-onnx` provides an **ONNX Runtime (ORT)** compute backend implementation for [GoMLX](https://github.com/gomlx/gomlx).

### Relationship to `gomlx/compute` and `gomlx/gomlx`

- **[`github.com/gomlx/gomlx`](https://github.com/gomlx/gomlx)**: The high-level Machine Learning framework for Go (tensors, layers, automatic differentiation, optimizers, model training loops). GoMLX delegates lower-level tensor graph compilation and execution to a `compute.Backend`.
- **[`github.com/gomlx/compute`](https://github.com/gomlx/compute)**: The foundational execution and backend abstraction interface (`compute.Backend`, `compute.Builder`, `compute.Function`, `compute.Executable`, `compute.Buffer`). It defines the unified IR (Intermediate Representation) graph and operator set that any hardware/framework execution engine must implement.
- **`github.com/gomlx/compute-onnx` (This Repository)**: Implements `compute.Backend` by translating `compute` computation graphs into ONNX Model Protocol Buffers (`onnx.ModelProto`) at runtime and executing them using ONNX Runtime via CGO bindings.

---

## Implementation Architecture & Execution Flow

1. **Graph Construction (`Builder` & `Function`)**:
   - As GoMLX builds a model or training step, it calls methods on `compute.Function` (e.g. `Add`, `MatMul`, `ConvGeneral`, `ReduceWindow`).
   - `compute-onnx` records these operations as a directed acyclic graph of `*Node` structures, mapping GoMLX computation nodes to corresponding ONNX operator definitions and attributes.

2. **Compilation (`compile.go`)**:
   - Upon `Builder.Compile()`, `compile.go` converts the GoMLX AST into an `onnx.ModelProto` (ONNX IR version 9, opset 21).
   - The serialized ONNX protobuf bytes are passed directly to ONNX Runtime via `ort.NewDynamicAdvancedSessionWithONNXData` to create an `ort.DynamicAdvancedSession`.

3. **Execution (`executable.go` & `buffer.go`)**:
   - **CPU Path**: Manages CPU tensor memory using `internal/pool` to re-use backing byte slices for inputs and outputs without allocation overhead.
   - **CUDA/GPU Path**: Uses ONNX Runtime `IoBinding` to keep tensors directly in GPU memory between sequential graph executions without copying data back to host CPU memory. It dynamically loads `libcudart.so` for zero-copy device transfers and manages a concurrent execution context pool (`cudaExecPool`).

---

## Package Directory Structure

### Public & Support Packages

- **`github.com/gomlx/compute-onnx` (Root)**:
  Contains the core `compute.Backend` implementation:
  - `backend.go`: Backend registration, device enumeration, and ONNX Runtime library initialization.
  - `builder.go` / `function.go`: Graph construction nodes and `compute.Builder` / `compute.Function` implementations.
  - `compile.go`: Translates GoMLX AST graph nodes into ONNX `ModelProto` protocol buffer definitions.
  - `executable.go`: Prepares input/output bindings and manages CPU pooling and CUDA `IoBinding` execution.
  - `buffer.go`: `compute.Buffer` implementation wrapping CPU host memory or CUDA device memory.
  - `ops_*.go`: Individual operator implementations mapping GoMLX ops (`ConvGeneral`, `DotGeneral`, `ReduceWindow`, `SelectAndScatterMax`, `Gather`, `Scatter`, etc.) to ONNX operators.
  - `gpu_detect_*.go`: Platform-specific GPU availability and CUDA library detection.

- **`cmd/onnxruntime_installer`**:
  CLI utility tool for downloading, extracting, and installing official pre-built ONNX Runtime shared libraries (`libonnxruntime.so`) and headers for CPU or CUDA execution environments.

- **`support/onnxruntime`**:
  Helper package providing convenient backend initialization utilities (e.g., `onnxruntime.New()`) for applications consuming the ONNX Runtime backend.

---

### Internal Packages (`internal/`)

- **`internal/ort`**:
  Low-level CGO wrapper for the ONNX Runtime C API (`onnxruntime_c_api.h`). Exposes `DynamicAdvancedSession`, `IoBinding`, `MemoryInfo`, `Value`, and CUDA dynamic memory copy procedures (`cuda_copy.go`).

- **`internal/pool`**:
  Memory arena and slice pooling utilities used by the CPU execution path to eliminate heap allocations for intermediate computation buffers during repeated training steps.

- **`internal/protos`**:
  Auto-generated Go Protocol Buffer bindings for ONNX specifications (`onnx-ml.pb.go`), compiled from official `.proto` definitions.

- **Internal Tools (`internal/cmd/`)**:
  - **`internal/cmd/update_c_ort`**: Code generation / maintenance script that downloads and updates C header files (`onnxruntime_c_api.h`) from upstream ONNX Runtime releases.
  - **`internal/cmd/update_protos`**: Code generation tool that fetches official ONNX `.proto` files and runs `protoc` to re-generate Go structs in `internal/protos`.
