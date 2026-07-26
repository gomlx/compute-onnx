//go:build linux

package onnxruntime

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/pkg/errors"
)

// HasNvidiaGPU tries to guess if there is an actual Nvidia GPU installed.
func HasNvidiaGPU() bool {
	return hasNvidiaGPU()
}

var hasNvidiaGPU = sync.OnceValue[bool](func() bool {
	matches, err := filepath.Glob("/dev/nvidia*")
	if err == nil && len(matches) > 0 {
		return true
	}

	// Execute the nvidia-smi command if present
	_, lookErr := exec.LookPath("nvidia-smi")
	if lookErr != nil {
		return false
	}
	cmd := exec.Command("nvidia-smi")
	output, cmdErr := cmd.CombinedOutput()
	if cmdErr != nil {
		return false
	}
	return strings.Contains(string(output), "NVIDIA-SMI")
})

func checkCUDAAndCUDNN() error {
	if runtime.GOARCH != "amd64" {
		return nil // skip on non-amd64 architectures
	}

	// 1. Check CUDA: try nvcc or libcudart.so
	_, nvccErr := exec.LookPath("nvcc")

	hasCudaLib := false
	cudaPaths := []string{
		"/usr/lib/x86_64-linux-gnu",
		"/usr/local/cuda/lib64",
		"/usr/local/lib",
		"/usr/lib",
	}
	for _, p := range cudaPaths {
		if _, err := os.Stat(filepath.Join(p, "libcudart.so")); err == nil {
			hasCudaLib = true
			break
		}
		if matches, _ := filepath.Glob(filepath.Join(p, "libcudart.so.*")); len(matches) > 0 {
			hasCudaLib = true
			break
		}
	}

	if nvccErr != nil && !hasCudaLib {
		return errors.Errorf(
			"CUDA Toolkit not found. It is required for the ONNX Runtime CUDA provider to work.\n" +
			"Pointers to install:\n" +
			"  - Official NVIDIA CUDA Installation Guide: https://developer.nvidia.com/cuda-downloads\n" +
			"  - Via package manager (Ubuntu/Debian): `sudo apt install nvidia-cuda-toolkit` (or similar for your distro)\n" +
			"  - Or use conda/mamba: `conda install -c nvidia cuda-toolkit`",
		)
	}

	// 2. Check cuDNN: search for libcudnn.so
	hasCudnnLib := false
	for _, p := range cudaPaths {
		if _, err := os.Stat(filepath.Join(p, "libcudnn.so")); err == nil {
			hasCudnnLib = true
			break
		}
		if matches, _ := filepath.Glob(filepath.Join(p, "libcudnn.so.*")); len(matches) > 0 {
			hasCudnnLib = true
			break
		}
	}

	if !hasCudnnLib {
		return errors.Errorf(
			"cuDNN library not found. It is required for the ONNX Runtime CUDA provider to work.\n" +
			"Pointers to install:\n" +
			"  - Official NVIDIA cuDNN Installation Guide: https://developer.nvidia.com/cudnn\n" +
			"  - Via package manager (Ubuntu/Debian): `sudo apt install libcudnn9-dev` or `sudo apt install libcudnn8-dev` (depending on CUDA version)\n" +
			"  - Or use conda/mamba: `conda install -c nvidia cudnn`",
		)
	}

	return nil
}
