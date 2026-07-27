# compute-onnx

**EXPERIMENTAL**

ONNX Runtime based compute backend for GoMLX.

It allows GoMLX models to be executed via ONNX Runtime using either CPU or CUDA (NVIDIA GPU).

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
