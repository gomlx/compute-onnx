// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/backendtest"
	"github.com/gomlx/compute/support/testutil"
	"k8s.io/klog/v2"
)

var (
	backend compute.Backend
)

func TestCompliance(t *testing.T) {
	backendtest.RunAll(t, backend, nil)
}

func init() {
	klog.InitFlags(nil)
}

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

func TestMain(m *testing.M) {
	flag.Parse()
	setup()
	code := m.Run()
	teardown()
	os.Exit(code)
}

func TestIota(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %+v", err)
	}
	defer b.Finalize()

	t.Run("2D Int8 axis 0", func(t *testing.T) {
		got, err := testutil.Exec1(b, nil, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.Iota(shapes.Make(dtypes.Int8, 2, 3), 0)
		})
		if err != nil {
			t.Fatalf("Iota axis 0 failed: %+v", err)
		}
		want := [][]int8{{0, 0, 0}, {1, 1, 1}}
		if ok, diff := testutil.IsInDelta(want, got, 0); !ok {
			t.Errorf("Mismatch:\n%s", diff)
		}
	})

	t.Run("2D Int8 axis 1", func(t *testing.T) {
		got, err := testutil.Exec1(b, nil, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.Iota(shapes.Make(dtypes.Int8, 2, 3), 1)
		})
		if err != nil {
			t.Fatalf("Iota axis 1 failed: %+v", err)
		}
		want := [][]int8{{0, 1, 2}, {0, 1, 2}}
		if ok, diff := testutil.IsInDelta(want, got, 0); !ok {
			t.Errorf("Mismatch:\n%s", diff)
		}
	})

	t.Run("1D Int32", func(t *testing.T) {
		got, err := testutil.Exec1(b, nil, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.Iota(shapes.Make(dtypes.Int32, 5), 0)
		})
		if err != nil {
			t.Fatalf("Iota 1D failed: %+v", err)
		}
		want := []int32{0, 1, 2, 3, 4}
		if ok, diff := testutil.IsInDelta(want, got, 0); !ok {
			t.Errorf("Mismatch:\n%s", diff)
		}
	})

	t.Run("Uint64 axis 0", func(t *testing.T) {
		got, err := testutil.Exec1(b, nil, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.Iota(shapes.Make(dtypes.Uint64, 3, 2), 0)
		})
		if err != nil {
			t.Fatalf("Iota Uint64 failed: %+v", err)
		}
		want := [][]uint64{{0, 0}, {1, 1}, {2, 2}}
		if ok, diff := testutil.IsInDelta(want, got, 0); !ok {
			t.Errorf("Mismatch:\n%s", diff)
		}
	})

	t.Run("3D Float32 axis 1", func(t *testing.T) {
		got, err := testutil.Exec1(b, nil, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.Iota(shapes.Make(dtypes.Float32, 2, 2, 2), 1)
		})
		if err != nil {
			t.Fatalf("Iota 3D failed: %+v", err)
		}
		want := [][][]float32{{{0, 0}, {1, 1}}, {{0, 0}, {1, 1}}}
		if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
			t.Errorf("Mismatch:\n%s", diff)
		}
	})
}

func TestMakeScalar(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %+v", err)
	}
	defer b.Finalize()

	gotFloat, err := testutil.Exec1(b, nil, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		return MakeScalar(f.(*Function), 42, dtypes.Float32)
	})
	if err != nil {
		t.Fatalf("MakeScalar Float32 failed: %+v", err)
	}
	if gotFloat != float32(42.0) {
		t.Errorf("Expected float32(42.0), got %v", gotFloat)
	}

	gotInt, err := testutil.Exec1(b, nil, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		return MakeScalar(f.(*Function), 100, dtypes.Int64)
	})
	if err != nil {
		t.Fatalf("MakeScalar Int64 failed: %+v", err)
	}
	if gotInt != int64(100) {
		t.Errorf("Expected int64(100), got %v", gotInt)
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
	invalidNode := &Node{
		opType: "InvalidOpNameThatDoesNotExistInONNX",
		inputs: []*Node{param.(*Node)},
		shape:  shapes.Make(dtypes.Float32, 2, 2),
	}
	fn.addNode(invalidNode)
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
		wantCuda          bool
		wantLog           int
		wantCustomLibPath string
		wantErr           bool
	}{
		{config: "cpu", wantCuda: false, wantLog: -1, wantCustomLibPath: ""},
		{config: "cuda", wantCuda: true, wantLog: -1, wantCustomLibPath: ""},
		{config: "cuda,log=2", wantCuda: true, wantLog: 1, wantCustomLibPath: ""},
		{config: "onnx:cpu", wantCuda: false, wantLog: -1, wantCustomLibPath: ""},
		{config: "onnxruntime:cuda", wantCuda: true, wantLog: -1, wantCustomLibPath: ""},
		{config: "onnx:cuda,log=2", wantCuda: true, wantLog: 1, wantCustomLibPath: ""},
		{config: "openxla:cuda", wantErr: true},
		{config: cpuLibPath, wantCuda: false, wantLog: -1, wantCustomLibPath: cpuLibPath},
		{config: cudaLibPath, wantCuda: HasNvidiaGPU(), wantLog: -1, wantCustomLibPath: cudaLibPath},
		{config: "cuda," + cpuLibPath, wantCuda: true, wantLog: -1, wantCustomLibPath: cpuLibPath},
		{config: "invalid_option_xyz", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.config, func(t *testing.T) {
			gotCuda, gotLog, gotPath, err := parseConfig(tt.config)
			if (err != nil) != tt.wantErr {
				t.Fatalf("parseConfig(%q) error = %v, wantErr %v", tt.config, err, tt.wantErr)
			}
			if !tt.wantErr {
				if gotCuda != tt.wantCuda {
					t.Errorf("gotCuda = %v, want %v", gotCuda, tt.wantCuda)
				}
				if gotLog != tt.wantLog {
					t.Errorf("gotLog = %v, want %v", gotLog, tt.wantLog)
				}
				if gotPath != tt.wantCustomLibPath {
					t.Errorf("gotCustomLibPath = %q, want %q", gotPath, tt.wantCustomLibPath)
				}
			}
		})
	}
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

	err := initializeORT(false, nonExistentPath)
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
