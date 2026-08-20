// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"flag"
	"os"
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
