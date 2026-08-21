// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"flag"
	"os"
	"testing"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/compute/gobackend"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
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

	// Test Float16 directly with float16.Float16 (if supported by backend)
	if b.Capabilities().DTypes[dtypes.Float16] {
		gotF16, err := testutil.Exec1(b, nil, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return MakeScalar(f.(*Function), float16.FromFloat32(3.5), dtypes.Float16)
		})
		if err != nil {
			t.Fatalf("MakeScalar Float16 from float16.Float16 failed: %+v", err)
		}
		if gotF16 != float16.FromFloat32(3.5) {
			t.Errorf("Expected float16(3.5), got %v", gotF16)
		}

		// Test Float16 from float32/float64 number
		gotF16FromFloat, err := testutil.Exec1(b, nil, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return MakeScalar(f.(*Function), 3.5, dtypes.Float16)
		})
		if err != nil {
			t.Fatalf("MakeScalar Float16 from float failed: %+v", err)
		}
		if gotF16FromFloat != float16.FromFloat32(3.5) {
			t.Errorf("Expected float16(3.5), got %v", gotF16FromFloat)
		}
	}

	// Test BFloat16 directly with bfloat16.BFloat16 (if supported by backend)
	if b.Capabilities().DTypes[dtypes.BFloat16] {
		gotBF16, err := testutil.Exec1(b, nil, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return MakeScalar(f.(*Function), bfloat16.FromFloat32(2.5), dtypes.BFloat16)
		})
		if err != nil {
			t.Fatalf("MakeScalar BFloat16 from bfloat16.BFloat16 failed: %+v", err)
		}
		if gotBF16 != bfloat16.FromFloat32(2.5) {
			t.Errorf("Expected bfloat16(2.5), got %v", gotBF16)
		}
	}
}
