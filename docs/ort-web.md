# ONNX Runtime Web (WASM & Browser) Backend

`github.com/gomlx/compute-onnx` supports compiling to WebAssembly (`GOOS=js GOARCH=wasm`) and executing GoMLX models directly inside web browsers using **ONNX Runtime Web (ORT Web)**.

This enables deploying high-performance ML models inside client-side web applications and Progressive Web Apps (PWAs) with zero server-side inference overhead.

---

## 1. Execution Providers (Accelerators)

When compiled for `js/wasm`, the backend configuration string format is:

```text
onnx:<provider>[,log=<severity>]
```

Supported execution providers:

| Provider | Config Options | Description |
| :--- | :--- | :--- |
| **WebGPU** | `onnx:webgpu`, `onnx:gpu` | Executes computation kernels via client GPU shaders using WebGPU. Best for large models and parallel batches. |
| **WASM (CPU)** | `onnx:wasm`, `onnx:cpu` | Executes via CPU WebAssembly using SIMD instructions. Low latency for small models and single-sample inference. |
| **WebNN** | `onnx:webnn` | Executes using browser Web Neural Network hardware acceleration (NPU/GPU/CPU). Experimental in Chromium browsers. |
| **WebGL** | `onnx:webgl` | Legacy GPU shader acceleration via WebGL. |
| **Default** | `onnx` (or empty) | **Auto-detects**: Uses **WebGPU** if an active GPU adapter is available; otherwise automatically defaults to **WASM (CPU)**. |

If an explicitly requested provider is unavailable in the user's browser (e.g. requesting `onnx:webgpu` or `onnx:webnn` without hardware/flag support), `compute.NewWithConfig(...)` returns a descriptive error rather than silently failing.

---

## 2. Configuration Options

### Log Severity (`log=<level>`)
Controls the verbosity of ONNX Runtime's internal logging:
- `log=0`: **Error only** (Default, quiet).
- `log=1`: **Warnings and Errors** (Displays graph partitioner and EP assignment warnings).
- `log=2`: **Info, Warnings, and Errors**.
- `log=3`: **Verbose / Debug**.

**Example:**
```bash
GOMLX_BACKEND="onnx:webgpu,log=1"
```

---

## 3. Serving & Loading ONNX Runtime Web

There are three ways to make the ONNX Runtime Web JavaScript and WebAssembly binary assets available to your application:

### Method 1: Automatic CDN Injection (Zero Configuration)
If `window.ort` is not already loaded on the page when your Go WASM binary runs (for instance, during automated test runners like `wasmbrowsertest`), `compute-onnx` will automatically:
1. Dynamically create and append a `<script>` tag loading `ort.min.js` from `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js`.
2. Configure `ort.env.wasm.wasmPaths` to fetch the required `.wasm` engine modules from the CDN on demand.

*Note: Requires internet access on the client machine.*

---

### Method 2: Script Tag in `index.html` (Online Web Applications)
For standard web applications, include the official ORT Web script tag in your HTML file before starting your Go WebAssembly program:

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>GoMLX WASM App</title>
    <!-- 1. Load ONNX Runtime Web from CDN -->
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js"></script>

    <!-- 2. Load Go WASM exec runtime -->
    <script src="wasm_exec.js"></script>
</head>
<body>
    <script>
        const go = new Go();
        WebAssembly.instantiateStreaming(fetch("main.wasm"), go.importObject).then((result) => {
            go.run(result.instance);
        });
    </script>
</body>
</html>
```

---

### Method 3: Self-Hosted / Offline Deployment (PWAs & Air-Gapped)
For offline PWAs, intranet deployments, or air-gapped web servers:

#### 1. Download distribution files using the installer tool:
```bash
go run github.com/gomlx/compute-onnx/cmd/onnxruntime_web_installer -dir=./static/ort
```

This downloads the complete set of required runtime assets:
- `ort.min.js`
- `ort-wasm.wasm`
- `ort-wasm-simd.wasm`
- `ort-wasm-threaded.wasm`
- `ort-wasm-simd-threaded.wasm`
- `ort-wasm-simd-threaded.jsep.wasm` (used for WebGPU)

#### 2. Host and configure local paths in `index.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <!-- 1. Load local static copy -->
    <script src="/static/ort/ort.min.js"></script>
    <script>
        // Point ONNX Runtime to the local static directory for .wasm binaries
        ort.env.wasm.wasmPaths = "/static/ort/";
    </script>

    <!-- 2. Load Go runtime -->
    <script src="wasm_exec.js"></script>
</head>
<body>
    <script>
        const go = new Go();
        WebAssembly.instantiateStreaming(fetch("main.wasm"), go.importObject).then((result) => {
            go.run(result.instance);
        });
    </script>
</body>
</html>
```

---

## 4. Go Code Example

```go
package main

import (
	"fmt"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/compute-onnx"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
)

func main() {
	// Initialize the backend (defaults to WebGPU if present, else WASM CPU)
	backend, err := compute.NewWithConfig("onnx")
	if err != nil {
		panic(err)
	}
	defer backend.Finalize()

	// Define a GoMLX execution graph
	exec := graph.MustNewExec(backend, func(x *graph.Node) *graph.Node {
		return graph.Add(x, graph.Scalar(x.Graph(), 10.0))
	})

	input := tensors.FromValue([]float32{1.0, 2.0, 3.0})
	output := exec.MustCall(input)[0]

	fmt.Println("Result:", output.Value()) // [11, 12, 13]
}
```

Compile to WASM:
```bash
GOOS=js GOARCH=wasm go build -o main.wasm main.go
```

---

## 5. Testing with `wasmbrowsertest`

To run automated unit tests and benchmarks inside a real browser:

```bash
# 1. Install wasmbrowsertest
go install github.com/agnivade/wasmbrowsertest@master

# 2. Run tests in headless browser (WASM CPU)
GOOS=js GOARCH=wasm go test -exec wasmbrowsertest ./...

# 3. Run tests in browser window with full WebGPU hardware acceleration
WASM_HEADLESS=off GOOS=js GOARCH=wasm go test -exec wasmbrowsertest ./...
```
