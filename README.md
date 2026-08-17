# compute-onnx

[![Documentation](https://img.shields.io/badge/docs-gomlx.github.io-blue.svg)](https://gomlx.github.io/)
[![Sponsor GoMLX](https://img.shields.io/badge/Sponsor-GoMLX-white?logo=github&style=flat-square)](https://github.com/gomlx/gomlx/blob/main/README.md#-support-the-project)

🚧🏗️ **EXPERIMENTAL** 🏗️🚧

ONNX Runtime based compute backend for GoMLX.

It allows GoMLX models to be executed via ONNX Runtime using either CPU or CUDA (NVIDIA GPU).

It supports dynamic shapes and exporting models to `.onnx` files.

## Example Usage

To run the Adult dataset demo with the ONNX backend:

```bash
GOMLX_BACKEND=onnx go run -tags=onnx github.com/gomlx/gomlx/examples/adult/demo
```

Or targeting a specific accelerator (e.g. CUDA):

```bash
GOMLX_BACKEND=onnx:cuda go run -tags=onnx github.com/gomlx/gomlx/examples/adult/demo
```

## Backend Options & Configuration

Configuration can be specified in the `GOMLX_BACKEND` environment variable using `onnx:<options>` or `onnxruntime:<options>` (comma-separated).

### Accelerator Selection

- **`cpu`**: Force CPU execution.
  ```bash
  GOMLX_BACKEND=onnx:cpu
  ```
- **`cuda`** / **`gpu`**: Force CUDA GPU execution (uses ONNX Runtime CUDA Execution Provider via `OrtIoBinding`).
  ```bash
  GOMLX_BACKEND=onnx:cuda
  ```
- **empty (default)**: Automatically detects if an NVIDIA GPU is present via `nvidia-smi` and defaults to CUDA if available, otherwise falling back to CPU.
  ```bash
  GOMLX_BACKEND=onnx
  ```

### Save Model To ONNX

This allows one to export GOMLX trained (or fine-tuned) models to ONNX.

See [an example in UCI-Adult demo](https://github.com/gomlx/gomlx/blob/main/examples/adult/demo/save_onnx.go). If you have a pre-trained file in a directory called `base`:

```
GOMLX_BACKEND=onnx go run -tags=onnx ./examples/adult/demo/ -checkpoint "base" -save_onnx="/tmp/a.onnx" -vmodule=save_onnx=1
```

### Inspecting `.onnx` Files (`onnx_printer`)

This repository includes a CLI tool in `cmd/onnx_printer` to inspect and pretty-print `.onnx` model files in the terminal:

```bash
go run github.com/gomlx/compute-onnx/cmd/onnx_printer path/to/model.onnx
```

It formats input, output, and node tensor shapes using GoMLX `shapes.Shape` (including named dynamic axes) and prints each graph operation on a single line. Tensor constants and initializers are truncated to 10 elements by default (controlled via `-max_items` / `-n`).

Example usage:
```bash
# Print model details with a maximum of 5 items for constant values
go run github.com/gomlx/compute-onnx/cmd/onnx_printer -max_items 5 /tmp/model.onnx

# Read from stdin
cat /tmp/model.onnx | go run github.com/gomlx/compute-onnx/cmd/onnx_printer
```

> **Tip**: For interactive graphical visualization of ONNX models, you can open `.onnx` model files using [Netron](https://netron.app/).


### Logging & Verbosity

- **Backend Log Level (`log=<level>`)**: Configures ONNX Runtime's internal logging severity level.
  - `log=0`: Errors only (severity level 3 / ERROR)
  - `log=1`: Warnings (severity level 2 / WARNING)
  - `log=2`: Informational (severity level 1 / INFO)
  - `log=3`: Verbose (severity level 0 / VERBOSE)
  
  Example:
  ```bash
  GOMLX_BACKEND="onnx:cuda,log=2"
  ```

- **Execution Timing Log (`-vmodule=executable=1`)**: Enables per-step execution timing breakdown printed via `klog` using `humanize.Duration`.
  
  Example:
  ```bash
  GOMLX_BACKEND=onnx:cuda go run -tags=onnx github.com/gomlx/gomlx/examples/adult/demo -vmodule=executable=1
  ```
