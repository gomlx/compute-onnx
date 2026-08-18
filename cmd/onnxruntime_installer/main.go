// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/gomlx/compute-onnx/support/onnxruntime"
)

var (
	flagVersion     = flag.String("version", onnxruntime.DefaultVersion, "The version of ONNX Runtime to install. Leave empty to install the latest version.")
	flagCuda        = flag.Bool("cuda", false, "Install the CUDA ONNX Runtime instead of the CPU one.")
	flagCudaVersion = flag.String("cuda_version", "", "The CUDA version to install for (e.g., '12', '13'). Defaults to auto-detected CUDA version or '12'.")
	flagTarget      = flag.String("target", "", "Target directory where to install the ONNX Runtime library. If empty, uses the default install location.")
	flagForce       = flag.Bool("force", false, "Force downloading and reinstalling the library even if already present.")
)

func main() {
	flag.Parse()

	libPath, err := onnxruntime.Install(*flagVersion, *flagCuda, *flagCudaVersion, *flagTarget, *flagForce)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error installing ONNX Runtime: %+v\n", err)
		os.Exit(1)
	}

	fmt.Printf("ONNX Runtime library successfully installed to:\n  %s\n", libPath)
}
