// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build linux

package onnxbackend

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestGetCUDALibrarySearchPaths(t *testing.T) {
	// Test environment variable parsing
	tmpDir := t.TempDir()
	customCuda := filepath.Join(tmpDir, "cuda_custom")
	customCudnn := filepath.Join(tmpDir, "cudnn_custom")
	customConda := filepath.Join(tmpDir, "conda_env")

	t.Setenv("CUDA_PATH", customCuda)
	t.Setenv("CUDNN_PATH", customCudnn)
	t.Setenv("CONDA_PREFIX", customConda)
	t.Setenv("LD_LIBRARY_PATH", filepath.Join(tmpDir, "ld_lib"))

	paths, _, _ := getCUDALibrarySearchPaths()

	expectedSubpaths := []string{
		filepath.Join(tmpDir, "ld_lib"),
		customCuda,
		filepath.Join(customCuda, "lib64"),
		filepath.Join(customCuda, "targets", "x86_64-linux", "lib"),
		customCudnn,
		filepath.Join(customCudnn, "lib64"),
		filepath.Join(customCudnn, "targets", "x86_64-linux", "lib"),
		filepath.Join(customConda, "lib"),
	}

	for _, expected := range expectedSubpaths {
		if !slices.Contains(paths, expected) {
			t.Errorf("expected path %q not found in search paths", expected)
		}
	}
}

func TestCheckCUDAAndCUDNNMock(t *testing.T) {
	// Test detection with mock libraries in custom directory
	tmpDir := t.TempDir()
	cudaLibDir := filepath.Join(tmpDir, "cuda", "lib64")
	cudnnLibDir := filepath.Join(tmpDir, "cudnn", "lib64")

	if err := os.MkdirAll(cudaLibDir, 0755); err != nil {
		t.Fatalf("failed to create dir: %v", err)
	}
	if err := os.MkdirAll(cudnnLibDir, 0755); err != nil {
		t.Fatalf("failed to create dir: %v", err)
	}

	// Create fake shared libraries
	if err := os.WriteFile(filepath.Join(cudaLibDir, "libcudart.so.12.0.0"), []byte("mock"), 0644); err != nil {
		t.Fatalf("failed to write mock libcudart: %v", err)
	}
	if err := os.WriteFile(filepath.Join(cudnnLibDir, "libcudnn.so.9.0.0"), []byte("mock"), 0644); err != nil {
		t.Fatalf("failed to write mock libcudnn: %v", err)
	}

	t.Setenv("CUDA_HOME", filepath.Join(tmpDir, "cuda"))
	t.Setenv("CUDNN_HOME", filepath.Join(tmpDir, "cudnn"))

	if err := checkCUDAAndCUDNN(); err != nil {
		t.Errorf("checkCUDAAndCUDNN failed with custom environment: %v", err)
	}
}
