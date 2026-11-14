// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"bytes"
	"slices"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

// TestONNXDynamicAndSerialization tests ONNX-specific behavior for dynamic shapes
// and constants (such as SaveModel/LoadModel round-tripping, AST constant caching,
// and fixed-size Go array constants).
//
// Generic dynamic-shape and operator tests live in
// github.com/gomlx/compute/support/backendtest (TestDynamicOps) and run via TestCompliance.
func TestONNXDynamicAndSerialization(t *testing.T) {
	if backend.Capabilities().DynamicShapes != compute.DynamicShapesNative {
		t.Fatalf("expected DynamicShapesNative capability, got %s", backend.Capabilities().DynamicShapes)
	}

	t.Run("ConstantCachingAndArray", func(t *testing.T) {
		builder := backend.Builder("test_const_cache")
		fn := builder.Main().(*Function)

		c1, err := fn.Constant([]int64{1, 2}, 2)
		if err != nil {
			t.Fatalf("Constant c1 failed: %v", err)
		}
		c2, err := fn.Constant([]int64{1, 2}, 2)
		if err != nil {
			t.Fatalf("Constant c2 failed: %v", err)
		}
		if c1 != c2 {
			t.Fatalf("expected c1 and c2 to be the same pointer")
		}

		p, err := fn.Parameter("x", shapes.Make(dtypes.Float32, 2, 2), nil)
		if err != nil {
			t.Fatalf("Parameter failed: %v", err)
		}
		add1, err := fn.Add(p, p)
		if err != nil {
			t.Fatalf("Add add1 failed: %v", err)
		}
		add2, err := fn.Add(p, p)
		if err != nil {
			t.Fatalf("Add add2 failed: %v", err)
		}
		if add1 == add2 {
			t.Fatalf("expected add1 and add2 to be distinct pointers")
		}

		// Fixed-size Go array constant support (used by some ONNX-GoMLX callers).
		b2 := backend.Builder("test_const_array")
		fn2 := b2.Main()
		arr := [3]float32{100, 200, 300}
		cArr, err := fn2.Constant(arr, 3)
		if err != nil {
			t.Fatalf("Constant([3]float32) failed: %v", err)
		}
		if err := fn2.Return([]compute.Value{cArr}, nil); err != nil {
			t.Fatalf("Return failed: %v", err)
		}
		ex, err := b2.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer ex.Finalize()
		outs, err := ex.Execute(nil, nil, 0)
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}
		defer outs[0].Finalize()
		got := make([]float32, 3)
		if err := outs[0].ToFlatData(got); err != nil {
			t.Fatalf("ToFlatData failed: %v", err)
		}
		if !slices.Equal(got, []float32{100, 200, 300}) {
			t.Fatalf("got %v, want [100 200 300]", got)
		}
	})

	t.Run("SaveAndLoadDynamicModel", func(t *testing.T) {
		onBackend := backend.(*Backend)
		onBackend.SetKeepModelProto(true)
		defer onBackend.SetKeepModelProto(false)

		builder := backend.Builder("test_save_dyn")
		mainFn := builder.Main()

		paramShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 3}, []string{"batch", ""})
		p, err := mainFn.Parameter("x", paramShape, nil)
		if err != nil {
			t.Fatalf("Parameter failed: %v", err)
		}
		doubleP, err := mainFn.Add(p, p)
		if err != nil {
			t.Fatalf("Add failed: %v", err)
		}
		if err := mainFn.Return([]compute.Value{doubleP}, nil); err != nil {
			t.Fatalf("Return failed: %v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer exec.Finalize()

		var buf bytes.Buffer
		if err := SaveModel(backend, exec, &buf, []string{"input_x"}, []string{"output_y"}); err != nil {
			t.Fatalf("SaveModel failed: %v", err)
		}

		loadedExec, err := LoadModel(backend, &buf)
		if err != nil {
			t.Fatalf("LoadModel failed: %v", err)
		}
		defer loadedExec.Finalize()

		inBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, shapes.Make(dtypes.Float32, 2, 3))
		if err != nil {
			t.Fatalf("BufferFromFlatData failed: %v", err)
		}
		defer inBuf.Finalize()

		outBufs, err := loadedExec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}
		if len(outBufs) != 1 {
			t.Fatalf("expected 1 output buffer, got %d", len(outBufs))
		}
		defer outBufs[0].Finalize()

		res := make([]float32, 6)
		if err := outBufs[0].ToFlatData(res); err != nil {
			t.Fatalf("ToFlatData failed: %v", err)
		}
		if !slices.Equal(res, []float32{2, 4, 6, 8, 10, 12}) {
			t.Fatalf("expected res [2 4 6 8 10 12], got %v", res)
		}
	})

	t.Run("AnonymousDynamicAxesSaveAndLoad", func(t *testing.T) {
		onBackend := backend.(*Backend)
		onBackend.SetKeepModelProto(true)
		defer onBackend.SetKeepModelProto(false)

		builder := backend.Builder("test_anon_dyn")
		mainFn := builder.Main()

		// Two independent anonymous axes at index 0 must not share a symbolic DimParam in ONNX.
		anonShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim}, []string{shapes.AnonymousAxis})
		p0, err := mainFn.Parameter("a", anonShape, nil)
		if err != nil {
			t.Fatalf("Parameter a failed: %v", err)
		}
		p1, err := mainFn.Parameter("b", anonShape, nil)
		if err != nil {
			t.Fatalf("Parameter b failed: %v", err)
		}
		concat, err := mainFn.Concatenate(0, p0, p1)
		if err != nil {
			t.Fatalf("Concatenate failed: %v", err)
		}
		if err := mainFn.Return([]compute.Value{concat}, nil); err != nil {
			t.Fatalf("Return failed: %v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer exec.Finalize()

		var buf bytes.Buffer
		if err := SaveModel(backend, exec, &buf, nil, nil); err != nil {
			t.Fatalf("SaveModel failed: %v", err)
		}
		loadedExec, err := LoadModel(backend, &buf)
		if err != nil {
			t.Fatalf("LoadModel failed: %v", err)
		}
		defer loadedExec.Finalize()

		inA, err := backend.BufferFromFlatData(0, []float32{1, 2}, shapes.Make(dtypes.Float32, 2))
		if err != nil {
			t.Fatalf("BufferFromFlatData a failed: %v", err)
		}
		defer inA.Finalize()
		inB, err := backend.BufferFromFlatData(0, []float32{3, 4, 5}, shapes.Make(dtypes.Float32, 3))
		if err != nil {
			t.Fatalf("BufferFromFlatData b failed: %v", err)
		}
		defer inB.Finalize()

		outBufs, err := loadedExec.Execute([]compute.Buffer{inA, inB}, nil, 0)
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}
		defer outBufs[0].Finalize()

		got := make([]float32, 5)
		if err := outBufs[0].ToFlatData(got); err != nil {
			t.Fatalf("ToFlatData failed: %v", err)
		}
		if !slices.Equal(got, []float32{1, 2, 3, 4, 5}) {
			t.Fatalf("expected [1 2 3 4 5], got %v", got)
		}
	})

	t.Run("FusedTopKSaveAndLoad", func(t *testing.T) {
		onBackend := backend.(*Backend)
		onBackend.SetKeepModelProto(true)
		defer onBackend.SetKeepModelProto(false)

		b := backend.Builder("topk_save_load")
		fn := b.Main()
		p, err := fn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, shapes.DynamicDim}, []string{"batch", "vocab"}), nil)
		if err != nil {
			t.Fatal(err)
		}
		_, idxs, err := fn.FusedTopK(p, 2, compute.FusedTopKConfig{Axis: 1})
		if err != nil {
			t.Fatal(err)
		}
		if err := fn.Return([]compute.Value{idxs}, nil); err != nil {
			t.Fatal(err)
		}
		ex, err := b.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer ex.Finalize()

		var modelBuf bytes.Buffer
		if err := SaveModel(backend, ex, &modelBuf, nil, nil); err != nil {
			t.Fatalf("SaveModel: %v", err)
		}
		loaded, err := LoadModel(backend, bytes.NewReader(modelBuf.Bytes()))
		if err != nil {
			t.Fatalf("LoadModel: %v", err)
		}
		defer loaded.Finalize()

		inBuf, err := backend.BufferFromFlatData(0, []float32{
			1, 5, 5, 2,
			9, 0, 4, 8,
			-3, -1, -2, -1,
		}, shapes.Make(dtypes.Float32, 3, 4))
		if err != nil {
			t.Fatal(err)
		}
		defer inBuf.Finalize()
		outs, err := loaded.Execute([]compute.Buffer{inBuf}, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer outs[0].Finalize()
		got := make([]int32, 6)
		if err := outs[0].ToFlatData(got); err != nil {
			t.Fatal(err)
		}
		want := []int32{1, 2, 0, 3, 1, 3}
		if !slices.Equal(got, want) {
			t.Fatalf("got %v, want %v", got, want)
		}
	})
}
