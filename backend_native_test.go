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
		wantEP            executionprovider.Type
		wantLog           int
		wantCustomLibPath string
		wantCacheDir      string
		wantSessionConfig SessionConfig
		wantErr           bool
	}{
		{config: "cpu", wantEP: executionprovider.CPU, wantLog: -1, wantCustomLibPath: ""},
		{config: "cuda", wantEP: executionprovider.CUDA, wantLog: -1, wantCustomLibPath: ""},
		{config: "cuda,log=2", wantEP: executionprovider.CUDA, wantLog: 1, wantCustomLibPath: ""},
		{config: "onnx:cpu", wantEP: executionprovider.CPU, wantLog: -1, wantCustomLibPath: ""},
		{config: "onnxruntime:cuda", wantEP: executionprovider.CUDA, wantLog: -1, wantCustomLibPath: ""},
		{config: "onnx:cuda,log=2", wantEP: executionprovider.CUDA, wantLog: 1, wantCustomLibPath: ""},
		{config: "migraphx", wantEP: executionprovider.MIGraphX, wantLog: -1, wantCustomLibPath: ""},
		{config: "rocm", wantEP: executionprovider.MIGraphX, wantLog: -1, wantCustomLibPath: ""},
		{config: "amd,log=0", wantEP: executionprovider.MIGraphX, wantLog: 3, wantCustomLibPath: ""},
		{config: "openxla:cuda", wantErr: true},
		{config: cpuLibPath, wantEP: executionprovider.CPU, wantLog: -1, wantCustomLibPath: cpuLibPath},
		// Auto-detection with a custom lib path depends on the GPUs present:
		{config: cudaLibPath, wantEP: autoDetectedEP(cudaDir), wantLog: -1, wantCustomLibPath: cudaLibPath},
		{config: "cuda," + cpuLibPath, wantEP: executionprovider.CUDA, wantLog: -1, wantCustomLibPath: cpuLibPath},
		{
			config: "onnx:cpu,intra_op_num_threads=1,inter_op_num_threads=1,cpu_mem_arena=false,execution_mode=parallel",
			wantEP: executionprovider.CPU,
			wantLog: -1,
			wantSessionConfig: func() SessionConfig {
				c := DefaultSessionConfig()
				c.IntraOpNumThreads = 1
				c.InterOpNumThreads = 1
				f := false
				c.CpuMemArena = &f
				c.ExecutionMode = "parallel"
				return c
			}(),
		},
		{
			config: "cpu,mem_pattern=false,graph_optimization_level=all",
			wantEP: executionprovider.CPU,
			wantLog: -1,
			wantSessionConfig: func() SessionConfig {
				c := DefaultSessionConfig()
				f := false
				c.MemPattern = &f
				c.GraphOptimizationLevel = 99
				return c
			}(),
		},
		{
			config: "cpu,session_clones=4",
			wantEP: executionprovider.CPU,
			wantLog: -1,
			wantSessionConfig: func() SessionConfig {
				c := DefaultSessionConfig()
				c.SessionClones = 4
				return c
			}(),
		},
		{
			config: "cpu,clones=12",
			wantEP: executionprovider.CPU,
			wantLog: -1,
			wantSessionConfig: func() SessionConfig {
				c := DefaultSessionConfig()
				c.SessionClones = 12
				return c
			}(),
		},
		{config: "cpu,session_clones=0", wantErr: true},
		{config: "cpu,session_clones=-1", wantErr: true},
		{config: "invalid_option_xyz", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.config, func(t *testing.T) {
			gotEP, gotLog, gotPath, gotCacheDir, gotSessionConfig, err := parseConfig(tt.config)
			if (err != nil) != tt.wantErr {
				t.Fatalf("parseConfig(%q) error = %v, wantErr %v", tt.config, err, tt.wantErr)
			}
			if !tt.wantErr {
				if gotEP != tt.wantEP {
					t.Errorf("got executionProvider = %v, want %v", gotEP, tt.wantEP)
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
				if tt.wantSessionConfig.IntraOpNumThreads != 0 || tt.wantSessionConfig.ExecutionMode != "" || tt.wantSessionConfig.CpuMemArena != nil || tt.wantSessionConfig.SessionClones > 0 {
					if gotSessionConfig.IntraOpNumThreads != tt.wantSessionConfig.IntraOpNumThreads {
						t.Errorf("got IntraOpNumThreads = %d, want %d", gotSessionConfig.IntraOpNumThreads, tt.wantSessionConfig.IntraOpNumThreads)
					}
					if gotSessionConfig.InterOpNumThreads != tt.wantSessionConfig.InterOpNumThreads {
						t.Errorf("got InterOpNumThreads = %d, want %d", gotSessionConfig.InterOpNumThreads, tt.wantSessionConfig.InterOpNumThreads)
					}
					if (gotSessionConfig.CpuMemArena == nil) != (tt.wantSessionConfig.CpuMemArena == nil) ||
						(gotSessionConfig.CpuMemArena != nil && *gotSessionConfig.CpuMemArena != *tt.wantSessionConfig.CpuMemArena) {
						t.Errorf("got CpuMemArena = %v, want %v", gotSessionConfig.CpuMemArena, tt.wantSessionConfig.CpuMemArena)
					}
					if gotSessionConfig.ExecutionMode != tt.wantSessionConfig.ExecutionMode {
						t.Errorf("got ExecutionMode = %q, want %q", gotSessionConfig.ExecutionMode, tt.wantSessionConfig.ExecutionMode)
					}
					if tt.wantSessionConfig.SessionClones > 0 && gotSessionConfig.SessionClones != tt.wantSessionConfig.SessionClones {
						t.Errorf("got SessionClones = %d, want %d", gotSessionConfig.SessionClones, tt.wantSessionConfig.SessionClones)
					}
				}
			}
		})
	}
}

// autoDetectedEP returns the expected GPU EP when no explicit provider token is given,
// given a directory that contains a fake CUDA provider library.
func autoDetectedEP(dir string) executionprovider.Type {
	if cuda.HasNvidiaGPU() && cuda.IsCUDALibraryAvailable(dir) {
		return executionprovider.CUDA
	}
	return executionprovider.CPU
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

	err := initializeORT(executionprovider.CPU, nonExistentPath)
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

func TestSessionOptionsEndToEnd(t *testing.T) {
	b, err := New("cpu,intra_op_num_threads=1,inter_op_num_threads=1,cpu_mem_arena=false,execution_mode=parallel,mem_pattern=false,graph_optimization_level=all")
	if err != nil {
		t.Fatalf("Failed to create CPU backend with session options: %+v", err)
	}
	defer b.Finalize()

	builder := b.Builder("test_session_options").(*Builder)
	fn := builder.Main().(*Function)
	param, err := fn.Parameter("x", shapes.Make(dtypes.Float32, 2, 2), nil)
	if err != nil {
		t.Fatalf("Failed to create parameter: %+v", err)
	}
	two, err := MakeScalar(fn, float32(2.0), dtypes.Float32)
	if err != nil {
		t.Fatalf("Failed to create scalar: %+v", err)
	}
	mulNode, err := fn.Mul(param.(*Node), two.(*Node))
	if err != nil {
		t.Fatalf("Failed to create Mul node: %+v", err)
	}
	fn.Return([]compute.Value{mulNode}, nil)

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Failed to compile graph with session options: %+v", err)
	}
	defer exec.Finalize()

	inputBuf, err := b.BufferFromFlatData(0, []float32{1.0, 2.0, 3.0, 4.0}, shapes.Make(dtypes.Float32, 2, 2))
	if err != nil {
		t.Fatalf("Failed to create input buffer: %+v", err)
	}
	defer inputBuf.Finalize()

	results, err := exec.Execute([]compute.Buffer{inputBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Failed to execute with session options: %+v", err)
	}
	defer results[0].Finalize()

	got := make([]float32, 4)
	if err := results[0].ToFlatData(got); err != nil {
		t.Fatalf("Failed to read result data: %+v", err)
	}
	want := []float32{2.0, 4.0, 6.0, 8.0}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("results[%d] = %v, want %v", i, got[i], w)
		}
	}
}

func TestConcurrentExecutionSessionClones(t *testing.T) {
	b, err := New("cpu,session_clones=4")
	if err != nil {
		t.Fatalf("Failed to create CPU backend: %+v", err)
	}
	defer b.Finalize()

	builder := b.Builder("test_concurrent_clones").(*Builder)
	fn := builder.Main().(*Function)
	param, err := fn.Parameter("x", shapes.Make(dtypes.Float32, 2), nil)
	if err != nil {
		t.Fatalf("Failed to create parameter: %+v", err)
	}
	two, err := MakeScalar(fn, float32(2.0), dtypes.Float32)
	if err != nil {
		t.Fatalf("Failed to create scalar: %+v", err)
	}
	mulNode, err := fn.Mul(param.(*Node), two.(*Node))
	if err != nil {
		t.Fatalf("Failed to create Mul node: %+v", err)
	}
	fn.Return([]compute.Value{mulNode}, nil)

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Failed to compile: %+v", err)
	}
	defer exec.Finalize()

	const numGoroutines = 16
	errChan := make(chan error, numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		go func(id int) {
			val := float32(id + 1)
			buf, err := b.BufferFromFlatData(0, []float32{val, val * 3}, shapes.Make(dtypes.Float32, 2))
			if err != nil {
				errChan <- err
				return
			}
			defer buf.Finalize()

			res, err := exec.Execute([]compute.Buffer{buf}, nil, 0)
			if err != nil {
				errChan <- err
				return
			}
			defer res[0].Finalize()

			got := make([]float32, 2)
			if err := res[0].ToFlatData(got); err != nil {
				errChan <- err
				return
			}
			if got[0] != val*2 || got[1] != val*6 {
				errChan <- fmt.Errorf("goroutine %d: got [%v, %v], want [%v, %v]", id, got[0], got[1], val*2, val*6)
				return
			}
			errChan <- nil
		}(g)
	}

	for g := 0; g < numGoroutines; g++ {
		if err := <-errChan; err != nil {
			t.Errorf("Concurrent execution error: %+v", err)
		}
	}
}

