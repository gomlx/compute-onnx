// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build linux

package onnxbackend

import (
	"bytes"
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

// getCUDALibrarySearchPaths returns candidate directories for CUDA and cuDNN libraries,
// along with boolean flags indicating if ldconfig already confirmed libcudart or libcudnn.
func getCUDALibrarySearchPaths() (paths []string, ldconfigCuda bool, ldconfigCudnn bool) {
	seen := make(map[string]bool)
	addPath := func(p string) {
		p = strings.TrimSpace(p)
		if p == "" {
			return
		}
		p = filepath.Clean(p)
		if !seen[p] {
			seen[p] = true
			paths = append(paths, p)
		}
	}

	// 1. Environment variables
	// LD_LIBRARY_PATH
	if ldLibPath := os.Getenv("LD_LIBRARY_PATH"); ldLibPath != "" {
		for _, dir := range filepath.SplitList(ldLibPath) {
			addPath(dir)
		}
	}

	// CUDA_PATH, CUDA_HOME, CUDA_ROOT
	for _, envVar := range []string{"CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"} {
		if val := os.Getenv(envVar); val != "" {
			addPath(val)
			addPath(filepath.Join(val, "lib64"))
			addPath(filepath.Join(val, "lib"))
			addPath(filepath.Join(val, "targets", "x86_64-linux", "lib"))
		}
	}

	// CUDNN_PATH, CUDNN_HOME, CUDNN_ROOT
	for _, envVar := range []string{"CUDNN_PATH", "CUDNN_HOME", "CUDNN_ROOT"} {
		if val := os.Getenv(envVar); val != "" {
			addPath(val)
			addPath(filepath.Join(val, "lib64"))
			addPath(filepath.Join(val, "lib"))
			addPath(filepath.Join(val, "targets", "x86_64-linux", "lib"))
		}
	}

	// CONDA_PREFIX, VIRTUAL_ENV
	for _, envVar := range []string{"CONDA_PREFIX", "VIRTUAL_ENV"} {
		if val := os.Getenv(envVar); val != "" {
			addPath(filepath.Join(val, "lib"))
			addPath(filepath.Join(val, "lib64"))
		}
	}

	// 2. Binary paths: Look for nvcc or PATH entries
	if nvccPath, err := exec.LookPath("nvcc"); err == nil {
		binDir := filepath.Dir(nvccPath)
		baseDir := filepath.Dir(binDir)
		addPath(filepath.Join(baseDir, "lib64"))
		addPath(filepath.Join(baseDir, "lib"))
		addPath(filepath.Join(baseDir, "targets", "x86_64-linux", "lib"))
	}

	if sysPath := os.Getenv("PATH"); sysPath != "" {
		for _, dir := range filepath.SplitList(sysPath) {
			cleanDir := filepath.Clean(dir)
			if filepath.Base(cleanDir) == "bin" {
				baseDir := filepath.Dir(cleanDir)
				addPath(filepath.Join(baseDir, "lib64"))
				addPath(filepath.Join(baseDir, "lib"))
				addPath(filepath.Join(baseDir, "targets", "x86_64-linux", "lib"))
			}
		}
	}

	// 3. Standard system & distribution library paths
	standardPaths := []string{
		"/usr/local/cuda/targets/x86_64-linux/lib",
		"/usr/local/cuda/lib64",
		"/usr/local/cuda/lib",
		"/usr/lib/x86_64-linux-gnu",
		"/usr/lib64",
		"/usr/lib",
		"/usr/local/lib64",
		"/usr/local/lib",
		"/opt/cuda/targets/x86_64-linux/lib",
		"/opt/cuda/lib64",
		"/opt/cuda/lib",
		"/usr/lib/cuda/lib64",
		"/usr/lib/cuda/lib",
		"/usr/lib/cuda",
		"/opt/nvidia/hpc_sdk/linux86-64/cuda/lib64",
	}
	for _, p := range standardPaths {
		addPath(p)
	}

	// Glob versioned installation patterns in /usr/local and /opt
	for _, pattern := range []string{
		"/usr/local/cuda-*/targets/x86_64-linux/lib",
		"/usr/local/cuda-*/lib64",
		"/usr/local/cuda-*/lib",
		"/opt/cuda-*/targets/x86_64-linux/lib",
		"/opt/cuda-*/lib64",
		"/opt/cuda-*/lib",
	} {
		if matches, err := filepath.Glob(pattern); err == nil {
			for _, m := range matches {
				addPath(m)
			}
		}
	}

	// 4. Query system dynamic linker cache via ldconfig -p
	if _, lookErr := exec.LookPath("ldconfig"); lookErr == nil {
		cmd := exec.Command("ldconfig", "-p")
		var out bytes.Buffer
		cmd.Stdout = &out
		if err := cmd.Run(); err == nil {
			lines := strings.Split(out.String(), "\n")
			for _, line := range lines {
				lineLower := strings.ToLower(line)
				if !strings.Contains(lineLower, "=>") {
					continue
				}

				isCuda := strings.Contains(lineLower, "libcudart.so")
				isCudnn := strings.Contains(lineLower, "libcudnn.so") || strings.Contains(lineLower, "libcudnn_")

				if isCuda {
					ldconfigCuda = true
				}
				if isCudnn {
					ldconfigCudnn = true
				}

				if isCuda || isCudnn {
					parts := strings.SplitN(line, "=>", 2)
					if len(parts) == 2 {
						targetPath := strings.TrimSpace(parts[1])
						if targetPath != "" {
							addPath(filepath.Dir(targetPath))
						}
					}
				}
			}
		}
	}

	return paths, ldconfigCuda, ldconfigCudnn
}

func checkCUDAAndCUDNN() error {
	if runtime.GOARCH != "amd64" {
		return nil // skip on non-amd64 architectures
	}

	cudaPaths, ldconfigCuda, ldconfigCudnn := getCUDALibrarySearchPaths()

	// 1. Check CUDA: try nvcc, ldconfig detection, or libcudart.so in search paths
	_, nvccErr := exec.LookPath("nvcc")
	hasCudaLib := ldconfigCuda

	if !hasCudaLib {
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
	}

	if nvccErr != nil && !hasCudaLib {
		return errors.Errorf(
			"CUDA Toolkit not found. It is required for the ONNX Runtime CUDA provider to work.\n" +
				"Pointers to install or locate:\n" +
				"  - Official NVIDIA CUDA Installation Guide: https://developer.nvidia.com/cuda-downloads\n" +
				"  - Via package manager (Ubuntu/Debian): `sudo apt install nvidia-cuda-toolkit` (or similar for your distro)\n" +
				"  - Or use conda/mamba: `conda install -c nvidia cuda-toolkit`\n" +
				"  - Set LD_LIBRARY_PATH or CUDA_PATH to point to your CUDA installation directory if it is installed in a non-standard location.",
		)
	}

	// 2. Check cuDNN: search for libcudnn.so or libcudnn_*.so in search paths, or ldconfig detection
	hasCudnnLib := ldconfigCudnn

	if !hasCudnnLib {
		for _, p := range cudaPaths {
			if _, err := os.Stat(filepath.Join(p, "libcudnn.so")); err == nil {
				hasCudnnLib = true
				break
			}
			if matches, _ := filepath.Glob(filepath.Join(p, "libcudnn.so.*")); len(matches) > 0 {
				hasCudnnLib = true
				break
			}
			if matches, _ := filepath.Glob(filepath.Join(p, "libcudnn_*.so*")); len(matches) > 0 {
				hasCudnnLib = true
				break
			}
		}
	}

	if !hasCudnnLib {
		return errors.Errorf(
			"cuDNN library not found. It is required for the ONNX Runtime CUDA provider to work.\n" +
				"Pointers to install or locate:\n" +
				"  - Official NVIDIA cuDNN Installation Guide: https://developer.nvidia.com/cudnn\n" +
				"  - Via package manager (Ubuntu/Debian): `sudo apt install libcudnn9-dev` or `sudo apt install libcudnn8-dev` (depending on CUDA version)\n" +
				"  - Or use conda/mamba: `conda install -c nvidia cudnn`\n" +
				"  - Set LD_LIBRARY_PATH or CUDNN_PATH (or CUDNN_HOME) to point to your cuDNN installation directory if it is installed in a non-standard location.",
		)
	}

	return nil
}
