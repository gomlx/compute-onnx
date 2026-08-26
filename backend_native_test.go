// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !js

package onnxbackend

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute-onnx/internal/device/cuda"
	"github.com/gomlx/compute-onnx/internal/executionprovider"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/backendtest"
	"k8s.io/klog/v2"
)

func setup() {
	fmt.Printf("Available backends: %q\n", compute.List())
	envVal := os.Getenv("GOMLX_BACKEND")
	config, err := ParseGOMLXBackendEnv(envVal)
	if err != nil {
		klog.Fatalf("Failed to parse GOMLX_BACKEND: %+v", err)
	}
	var errNew error
	backend, errNew = New(config)
	if errNew != nil {
		klog.Fatalf("Failed to create backend: %+v", errNew)
	}
	fmt.Printf("Backend: %s, %s\n", backend.Name(), backend.Description())
}

func teardown() {
	if backend != nil {
		backend.Finalize()
	}
}

func TestSaveOnFailureEnv(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %+v", err)
	}
	defer b.Finalize()

	tempDir := t.TempDir()
	savePath := filepath.Join(tempDir, "failed_model_test.onnx")
	t.Setenv(SaveOnFailureEnv, savePath)

	builder := b.Builder("test_failure").(*Builder)
	fn := builder.Main().(*Function)
	param, err := fn.Parameter("x", shapes.Make(dtypes.Float32, 2, 2), nil)
	if err != nil {
		t.Fatalf("Failed to create parameter: %+v", err)
	}
	invalidNode := fn.AddCustomNode("InvalidOpNameThatDoesNotExistInONNX", []*Node{param.(*Node)}, shapes.Make(dtypes.Float32, 2, 2))
	fn.Return([]compute.Value{invalidNode}, nil)

	_, compileErr := builder.Compile()
	if compileErr == nil {
		t.Fatal("Expected compilation to fail for invalid op, but it succeeded")
	}

	if _, statErr := os.Stat(savePath); os.IsNotExist(statErr) {
		t.Errorf("Expected failed model to be saved at %q, but file does not exist", savePath)
	}
}

func TestParseConfig(t *testing.T) {
	tempDir := t.TempDir()
	cpuLibPath := filepath.Join(tempDir, "libonnxruntime.so")
	_ = os.WriteFile(cpuLibPath, []byte("fake so"), 0644)

	cudaDir := filepath.Join(tempDir, "cuda_ort")
	_ = os.MkdirAll(cudaDir, 0755)
	cudaLibPath := filepath.Join(cudaDir, "libonnxruntime.so")
	_ = os.WriteFile(cudaLibPath, []byte("fake so"), 0644)
	_ = os.WriteFile(filepath.Join(cudaDir, "libonnxruntime_providers_cuda.so"), []byte("fake cuda provider so"), 0644)

	tests := []struct {
		config            string
		wantEP            executionprovider.ExecutionProviderType
		wantLog           int
		wantCustomLibPath string
		wantCacheDir      string
		wantErr           bool
	}{
		{config: "cpu", wantEP: executionprovider.ExecutionProviderCPU, wantLog: -1, wantCustomLibPath: ""},
		{config: "cuda", wantEP: executionprovider.ExecutionProviderCUDA, wantLog: -1, wantCustomLibPath: ""},
		{config: "cuda,log=2", wantEP: executionprovider.ExecutionProviderCUDA, wantLog: 1, wantCustomLibPath: ""},
		{config: "onnx:cpu", wantEP: executionprovider.ExecutionProviderCPU, wantLog: -1, wantCustomLibPath: ""},
		{config: "onnxruntime:cuda", wantEP: executionprovider.ExecutionProviderCUDA, wantLog: -1, wantCustomLibPath: ""},
		{config: "onnx:cuda,log=2", wantEP: executionprovider.ExecutionProviderCUDA, wantLog: 1, wantCustomLibPath: ""},
		{config: "migraphx", wantEP: executionprovider.ExecutionProviderMIGraphX, wantLog: -1, wantCustomLibPath: ""},
		{config: "rocm", wantEP: executionprovider.ExecutionProviderMIGraphX, wantLog: -1, wantCustomLibPath: ""},
		{config: "amd,log=0", wantEP: executionprovider.ExecutionProviderMIGraphX, wantLog: 3, wantCustomLibPath: ""},
		{config: "openxla:cuda", wantErr: true},
		{config: cpuLibPath, wantEP: executionprovider.ExecutionProviderCPU, wantLog: -1, wantCustomLibPath: cpuLibPath},
		// Auto-detection with a custom lib path depends on the GPUs present:
		{config: cudaLibPath, wantEP: autoDetectedEP(cudaDir), wantLog: -1, wantCustomLibPath: cudaLibPath},
		{config: "cuda," + cpuLibPath, wantEP: executionprovider.ExecutionProviderCUDA, wantLog: -1, wantCustomLibPath: cpuLibPath},
		{config: "invalid_option_xyz", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.config, func(t *testing.T) {
			gotEP, gotLog, gotPath, gotCacheDir, err := parseConfig(tt.config)
			if (err != nil) != tt.wantErr {
				t.Fatalf("parseConfig(%q) error = %v, wantErr %v", tt.config, err, tt.wantErr)
			}
			if !tt.wantErr {
				if gotEP != tt.wantEP {
					t.Errorf("got gpuEP = %v, want %v", gotEP, tt.wantEP)
				}
				if gotLog != tt.wantLog {
					t.Errorf("gotLog = %v, want %v", gotLog, tt.wantLog)
				}
				if gotPath != tt.wantCustomLibPath {
					t.Errorf("gotCustomLibPath = %q, want %q", gotPath, tt.wantCustomLibPath)
				}
				if gotCacheDir != tt.wantCacheDir {
					t.Errorf("gotMigraphxCacheDir = %q, want %q", gotCacheDir, tt.wantCacheDir)
				}
			}
		})
	}
}

// autoDetectedEP returns the expected GPU EP when no explicit provider token is given,
// given a directory that contains a fake CUDA provider library.
func autoDetectedEP(dir string) executionprovider.ExecutionProviderType {
	if cuda.HasNvidiaGPU() && cuda.IsCUDALibraryAvailable(dir) {
		return executionprovider.ExecutionProviderCUDA
	}
	return executionprovider.ExecutionProviderCPU
}

func TestEnableAutoInstall(t *testing.T) {
	EnableAutoInstall(false)
	if autoInstall != false {
		t.Errorf("expected autoInstall to be false after EnableAutoInstall(false)")
	}
	EnableAutoInstall(true)
	if autoInstall != true {
		t.Errorf("expected autoInstall to be true after EnableAutoInstall(true)")
	}
}

func TestExplicitPathNoAutoInstall(t *testing.T) {
	initMutex.Lock()
	wasInitialized := isOrtInitialized
	isOrtInitialized = false
	initMutex.Unlock()

	defer func() {
		initMutex.Lock()
		isOrtInitialized = wasInitialized
		initMutex.Unlock()
	}()

	EnableAutoInstall(true)
	nonExistentPath := filepath.Join(t.TempDir(), "nonexistent_libonnxruntime.so")

	err := initializeORT(executionprovider.ExecutionProviderCPU, nonExistentPath)
	if err == nil {
		t.Fatal("expected error when explicit path does not exist, got nil")
	}
	if !strings.Contains(err.Error(), "ONNX Runtime library not found at specified path") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestConvGeneralFloat64(t *testing.T) {
	t.Run("CPU", func(t *testing.T) {
		b, err := New("cpu")
		if err != nil {
			t.Fatalf("Failed to create CPU backend: %+v", err)
		}
		defer b.Finalize()
		backendtest.TestConvGeneral(t, b, nil)
	})

	t.Run("CUDA", func(t *testing.T) {
		b, err := New("cuda")
		if err != nil {
			t.Fatalf("Failed to create CUDA backend: %+v", err)
		}
		defer b.Finalize()
		backendtest.TestConvGeneral(t, b, nil)
	})
}
