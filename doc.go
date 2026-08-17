// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

/*
Package onnxbackend implements an ONNX Runtime (ORT) compute backend for GoMLX (github.com/gomlx/compute).

It allows GoMLX models to be executed via ONNX Runtime using either CPU or CUDA (NVIDIA GPU).
It supports dynamic shapes and exporting trained models to standard .onnx model files.

# Registration & Initialization

Importing this package registers the backend under the names "onnxruntime" and "onnx" with the GoMLX compute registry:

	import _ "github.com/gomlx/compute-onnx"

Once imported, the backend can be selected automatically by `compute.New()` using the GOMLX_BACKEND environment variable:

	GOMLX_BACKEND=onnx go run github.com/gomlx/gomlx/examples/adult/demo

Or created programmatically using `compute.NewWithConfig("onnx")` or directly via [New]:

	backend, err := onnxbackend.New("cuda,log=2")

# Configuration Options

Configuration options can be specified in the GOMLX_BACKEND environment variable as "onnx:<options>" (or "onnxruntime:<options>") or passed directly to [New]:

Accelerator Selection:
  - cpu: Force CPU execution.
    GOMLX_BACKEND=onnx:cpu
  - cuda / gpu: Force CUDA GPU execution using ONNX Runtime CUDA Execution Provider via IoBinding.
    GOMLX_BACKEND=onnx:cuda
  - (empty / default): Automatically detects if an NVIDIA GPU is available and defaults to CUDA, falling back to CPU otherwise.
    GOMLX_BACKEND=onnx

ONNX Runtime Internal Logging:
  - log=<level>: Sets ONNX Runtime internal logging severity level:
    - log=0: Errors only (ERROR)
    - log=1: Warnings (WARNING)
    - log=2: Informational (INFO)
    - log=3: Verbose (VERBOSE)

Example:

	GOMLX_BACKEND="onnx:cuda,log=2"

# ONNX Runtime Shared Libraries & Auto-Installation

The backend automatically locates or manages the required ONNX Runtime shared library (libonnxruntime.so / onnxruntime.dll):

  - Custom Library Path: Set the ONNXRUNTIME_SHARED_LIBRARY_PATH environment variable to point directly to the shared library binary.
  - Auto-Installation: If no library path is provided, the backend automatically downloads and extracts prebuilt official ONNX Runtime binaries locally (e.g. ~/.local/lib/onnxruntime/ on Linux).
  - Disabling Auto-Installation: Set the environment variable GOMLX_NO_AUTO_INSTALL=1 (or [NoAutoInstallEnv]) to disable automatic downloads (useful for offline environments or container deployments).

# Debugging & Saving Models on Failure

If graph compilation or session creation fails, setting the environment variable GOMLX_ONNX_SAVE_ON_FAILURE (or [SaveOnFailureEnv]) to a file path instructs the ONNX backend to save the serialized ONNX model protobuf bytes to that file path for debugging and log a notification:

	GOMLX_ONNX_SAVE_ON_FAILURE="/tmp/failed_model.onnx"

# Inspecting & Pretty-Printing `.onnx` Files

The CLI tool in `github.com/gomlx/compute-onnx/cmd/onnx_printer` pretty-prints the contents of `.onnx` model files in the terminal, displaying model metadata, input/output tensors with GoMLX shapes ([shapes.Shape]), initializers/constants, and graph operations on a single line per op:

	go run github.com/gomlx/compute-onnx/cmd/onnx_printer path/to/model.onnx

Flags:
  - max_items / n: Controls the maximum number of constant array elements printed (default 10).
  - show_doc: Includes docstrings if present.

For interactive graphical visualization of ONNX models, `.onnx` files can also be opened using Netron (https://netron.app/).
*/
package onnxbackend
