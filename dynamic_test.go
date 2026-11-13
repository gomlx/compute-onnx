// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"bytes"
	"slices"
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
)

func TestDynamicOps(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer backend.Finalize()

	if backend.Capabilities().DynamicShapes != compute.DynamicShapesNative {
		t.Fatalf("expected DynamicSupportNative capability, got %s", backend.Capabilities().DynamicShapes)
	}

	t.Run("DynamicShapeAndSize", func(t *testing.T) {
		builder := backend.Builder("test_dyn_shape")
		mainFn := builder.Main()

		paramShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 4}, []string{"batch", ""})
		x, err := mainFn.Parameter("x", paramShape, nil)
		if err != nil {
			t.Fatalf("Parameter failed: %v", err)
		}

		shapeVal, err := mainFn.DynamicShape(x)
		if err != nil {
			t.Fatalf("DynamicShape failed: %v", err)
		}

		dim0, err := mainFn.DynamicDimensionSize(x, 0)
		if err != nil {
			t.Fatalf("DynamicDimensionSize 0 failed: %v", err)
		}

		dim1, err := mainFn.DynamicDimensionSize(x, 1)
		if err != nil {
			t.Fatalf("DynamicDimensionSize 1 failed: %v", err)
		}

		err = mainFn.Return([]compute.Value{shapeVal, dim0, dim1}, nil)
		if err != nil {
			t.Fatalf("Return failed: %v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer exec.Finalize()

		inBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6, 7, 8}, shapes.Make(dtypes.Float32, 2, 4))
		if err != nil {
			t.Fatalf("BufferFromFlatData failed: %v", err)
		}
		defer inBuf.Finalize()

		outputs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}
		if len(outputs) != 3 {
			t.Fatalf("expected 3 outputs, got %d", len(outputs))
		}

		shapeData := make([]int64, 2)
		if err := outputs[0].ToFlatData(shapeData); err != nil {
			t.Fatalf("ToFlatData failed: %v", err)
		}
		if !slices.Equal(shapeData, []int64{2, 4}) {
			t.Fatalf("expected shape [2, 4], got %v", shapeData)
		}

		d0Data := make([]int64, 1)
		if err := outputs[1].ToFlatData(d0Data); err != nil {
			t.Fatalf("ToFlatData failed: %v", err)
		}
		if d0Data[0] != 2 {
			t.Fatalf("expected dim0=2, got %d", d0Data[0])
		}

		d1Data := make([]int64, 1)
		if err := outputs[2].ToFlatData(d1Data); err != nil {
			t.Fatalf("ToFlatData failed: %v", err)
		}
		if d1Data[0] != 4 {
			t.Fatalf("expected dim1=4, got %d", d1Data[0])
		}
	})

	t.Run("DynamicReshape", func(t *testing.T) {
		builder := backend.Builder("test_dyn_reshape")
		mainFn := builder.Main()

		paramShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, shapes.DynamicDim}, []string{"batch", "seq"})
		x, err := mainFn.Parameter("x", paramShape, nil)
		if err != nil {
			t.Fatalf("Parameter failed: %v", err)
		}

		batchSize, err := mainFn.DynamicDimensionSize(x, 0)
		if err != nil {
			t.Fatalf("DynamicDimensionSize 0 failed: %v", err)
		}

		seqLen, err := mainFn.DynamicDimensionSize(x, 1)
		if err != nil {
			t.Fatalf("DynamicDimensionSize 1 failed: %v", err)
		}

		numTokens, err := mainFn.Mul(batchSize, seqLen)
		if err != nil {
			t.Fatalf("Mul failed: %v", err)
		}

		reshaped, err := mainFn.DynamicReshape(x, compute.DynamicDimensionSpec{
			Name:  "num_tokens",
			Value: numTokens,
		})
		if err != nil {
			t.Fatalf("DynamicReshape failed: %v", err)
		}

		err = mainFn.Return([]compute.Value{reshaped}, nil)
		if err != nil {
			t.Fatalf("Return failed: %v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer exec.Finalize()

		inBuf, err := backend.BufferFromFlatData(0, []float32{10, 20, 30, 40, 50, 60}, shapes.Make(dtypes.Float32, 2, 3))
		if err != nil {
			t.Fatalf("BufferFromFlatData failed: %v", err)
		}
		defer inBuf.Finalize()

		outputs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}
		if len(outputs) != 1 {
			t.Fatalf("expected 1 output, got %d", len(outputs))
		}

		outData := make([]float32, 6)
		if err := outputs[0].ToFlatData(outData); err != nil {
			t.Fatalf("ToFlatData failed: %v", err)
		}
		if !slices.Equal(outData, []float32{10, 20, 30, 40, 50, 60}) {
			t.Fatalf("expected outData [10 20 30 40 50 60], got %v", outData)
		}
		outShape, err := outputs[0].Shape()
		if err != nil {
			t.Fatalf("Shape failed: %v", err)
		}
		if !slices.Equal(outShape.Dimensions, []int{6}) {
			t.Fatalf("expected outShape [6], got %v", outShape.Dimensions)
		}
	})

	t.Run("DynamicBroadcastInDim", func(t *testing.T) {
		builder := backend.Builder("test_dyn_broadcast")
		mainFn := builder.Main().(*Function)

		paramShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 1}, []string{"batch", ""})
		x, err := mainFn.Parameter("x", paramShape, nil)
		if err != nil {
			t.Fatalf("Parameter failed: %v", err)
		}

		batchSize, err := mainFn.DynamicDimensionSize(x, 0)
		if err != nil {
			t.Fatalf("DynamicDimensionSize failed: %v", err)
		}

		c, err := mainFn.Constant([]float32{10, 20, 30}, 1, 3)
		if err != nil {
			t.Fatalf("Constant failed: %v", err)
		}

		broadcasted, err := mainFn.DynamicBroadcastInDim(c, []int{0, 1},
			compute.DynamicDimensionSpec{Name: "batch", Value: batchSize},
			compute.DynamicDimensionSpec{Static: 3},
		)
		if err != nil {
			t.Fatalf("DynamicBroadcastInDim failed: %v", err)
		}

		err = mainFn.Return([]compute.Value{broadcasted}, nil)
		if err != nil {
			t.Fatalf("Return failed: %v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer exec.Finalize()

		inBuf, err := backend.BufferFromFlatData(0, []float32{1, 2}, shapes.Make(dtypes.Float32, 2, 1))
		if err != nil {
			t.Fatalf("BufferFromFlatData failed: %v", err)
		}
		defer inBuf.Finalize()

		outputs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}
		if len(outputs) != 1 {
			t.Fatalf("expected 1 output, got %d", len(outputs))
		}

		outData := make([]float32, 6)
		if err := outputs[0].ToFlatData(outData); err != nil {
			t.Fatalf("ToFlatData failed: %v", err)
		}
		wantData := []float32{10, 20, 30, 10, 20, 30}
		if !slices.Equal(outData, wantData) {
			t.Fatalf("expected outData %v, got %v", wantData, outData)
		}
	})

	t.Run("ConstantCaching", func(t *testing.T) {
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

		err = mainFn.Return([]compute.Value{doubleP}, nil)
		if err != nil {
			t.Fatalf("Return failed: %v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer exec.Finalize()

		var buf bytes.Buffer
		err = SaveModel(backend, exec, &buf, []string{"input_x"}, []string{"output_y"})
		if err != nil {
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

		res := make([]float32, 6)
		if err := outBufs[0].ToFlatData(res); err != nil {
			t.Fatalf("ToFlatData failed: %v", err)
		}
		if !slices.Equal(res, []float32{2, 4, 6, 8, 10, 12}) {
			t.Fatalf("expected res [2 4 6 8 10 12], got %v", res)
		}
	})

	t.Run("DynamicIota", func(t *testing.T) {
		builder := backend.Builder("test_dynamic_iota")
		mainFn := builder.Main()

		batchParam, err := mainFn.Parameter("batch", shapes.Make(dtypes.Int32), nil)
		if err != nil {
			t.Fatalf("Parameter failed: %v", err)
		}

		iotaVal, err := mainFn.DynamicIota(dtypes.Int32, 1,
			compute.DynamicDimensionSpec{Name: "batch", Value: batchParam},
			compute.DynamicDimensionSpec{Static: 3},
		)
		if err != nil {
			t.Fatalf("DynamicIota failed: %v", err)
		}

		err = mainFn.Return([]compute.Value{iotaVal}, nil)
		if err != nil {
			t.Fatalf("Return failed: %v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer exec.Finalize()

		inBuf, err := backend.BufferFromFlatData(0, []int32{2}, shapes.Make(dtypes.Int32))
		if err != nil {
			t.Fatalf("BufferFromFlatData failed: %v", err)
		}
		defer inBuf.Finalize()

		outBufs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}
		defer outBufs[0].Finalize()

		res := make([]int32, 6)
		if err := outBufs[0].ToFlatData(res); err != nil {
			t.Fatalf("ToFlatData failed: %v", err)
		}
		if !slices.Equal(res, []int32{0, 1, 2, 0, 1, 2}) {
			t.Fatalf("expected [0 1 2 0 1 2], got %v", res)
		}
	})

	t.Run("DynamicPad", func(t *testing.T) {
		builder := backend.Builder("test_dynamic_pad")
		mainFn := builder.Main()

		xParam, err := mainFn.Parameter("x", shapes.Make(dtypes.Float32, 2, 2), nil)
		if err != nil {
			t.Fatalf("Parameter x failed: %v", err)
		}
		padStartParam, err := mainFn.Parameter("pad_start", shapes.Make(dtypes.Int32), nil)
		if err != nil {
			t.Fatalf("Parameter pad_start failed: %v", err)
		}
		fillVal, err := mainFn.Constant([]float32{0})
		if err != nil {
			t.Fatalf("Constant failed: %v", err)
		}

		padded, err := mainFn.DynamicPad(xParam, fillVal,
			compute.DynamicPadAxis{StartValue: padStartParam, End: 1},
			compute.DynamicPadAxis{Start: 1, End: 0},
		)
		if err != nil {
			t.Fatalf("DynamicPad failed: %v", err)
		}

		err = mainFn.Return([]compute.Value{padded}, nil)
		if err != nil {
			t.Fatalf("Return failed: %v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer exec.Finalize()

		inX, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shapes.Make(dtypes.Float32, 2, 2))
		if err != nil {
			t.Fatalf("BufferFromFlatData x failed: %v", err)
		}
		defer inX.Finalize()

		inPad, err := backend.BufferFromFlatData(0, []int32{1}, shapes.Make(dtypes.Int32))
		if err != nil {
			t.Fatalf("BufferFromFlatData pad failed: %v", err)
		}
		defer inPad.Finalize()

		outBufs, err := exec.Execute([]compute.Buffer{inX, inPad}, []bool{false, false}, 0)
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}
		defer outBufs[0].Finalize()

		// output shape is (1+2+1, 1+2+0) = (4, 3), total 12 elements
		res := make([]float32, 12)
		if err := outBufs[0].ToFlatData(res); err != nil {
			t.Fatalf("ToFlatData failed: %v", err)
		}
		expected := []float32{
			0, 0, 0,
			0, 1, 2,
			0, 3, 4,
			0, 0, 0,
		}
		if !slices.Equal(res, expected) {
			t.Fatalf("expected %v, got %v", expected, res)
		}
	})

	t.Run("AnonymousDynamicAxesAndSaveLoad", func(t *testing.T) {
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

	t.Run("PoolAndArgMaxDynamic", func(t *testing.T) {
		builder := backend.Builder("test_pool_argmax_dyn")
		mainFn := builder.Main()

		inShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 4, 1}, []string{"batch", "", ""})
		x, err := mainFn.Parameter("x", inShape, nil)
		if err != nil {
			t.Fatalf("Parameter failed: %v", err)
		}
		pooled, err := mainFn.ReduceWindow(x, compute.ReduceOpMax, []int{1, 2, 1}, []int{1, 2, 1}, nil, nil, nil)
		if err != nil {
			t.Fatalf("ReduceWindow failed: %v", err)
		}
		argMax, err := mainFn.ArgMinMax(pooled, 1, dtypes.Int32, false)
		if err != nil {
			t.Fatalf("ArgMinMax failed: %v", err)
		}
		if err := mainFn.Return([]compute.Value{argMax}, nil); err != nil {
			t.Fatalf("Return failed: %v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer exec.Finalize()

		inBuf, err := backend.BufferFromFlatData(0, []float32{1, 5, 3, 2, 9, 4, 7, 8}, shapes.Make(dtypes.Float32, 2, 4, 1))
		if err != nil {
			t.Fatalf("BufferFromFlatData failed: %v", err)
		}
		defer inBuf.Finalize()

		outBufs, err := exec.Execute([]compute.Buffer{inBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}
		defer outBufs[0].Finalize()

		got := make([]int32, 2)
		if err := outBufs[0].ToFlatData(got); err != nil {
			t.Fatalf("ToFlatData failed: %v", err)
		}
		if !slices.Equal(got, []int32{0, 0}) {
			t.Fatalf("expected [0 0], got %v", got)
		}
	})

	t.Run("NonZeroAndUniqueAxisNames", func(t *testing.T) {
		// 1. Verify minted unique axis names on NonZero, ConvGeneral, and DotGeneral allow downstream binary/broadcast/iota ops.
		builder := backend.Builder("test_nonzero_unique_axis")
		mainFn := builder.Main()

		xShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 3}, []string{"rows", ""})
		x, err := mainFn.Parameter("x", xShape, nil)
		if err != nil {
			t.Fatalf("Parameter failed: %v", err)
		}

		coords, err := mainFn.NonZero(x, compute.NonZeroConfig{})
		if err != nil {
			t.Fatalf("NonZero failed: %v", err)
		}
		coordsShape, err := mainFn.Shape(coords)
		if err != nil {
			t.Fatalf("Shape(coords) failed: %v", err)
		}
		countAxis := coordsShape.AxisName(0)
		if !strings.HasPrefix(countAxis, "#nonzeros_") {
			t.Fatalf("expected minted #nonzeros_* axis name, got %q", countAxis)
		}

		// Duplicate explicit CountAxisName must fail.
		if _, err := mainFn.NonZero(x, compute.NonZeroConfig{CountAxisName: countAxis}); err == nil {
			t.Fatalf("expected error on duplicate CountAxisName %q", countAxis)
		}
		if _, err := mainFn.NonZero(x, compute.NonZeroConfig{CountAxisName: "rows"}); err == nil {
			t.Fatalf("expected error on CountAxisName conflicting with parameter axis")
		}

		// Downstream ops referencing the minted countAxis name without explicit Value node:
		rowIds, err := mainFn.DynamicIota(dtypes.Int64, 0,
			compute.DynamicDimensionSpec{Name: countAxis},
			compute.DynamicDimensionSpec{Static: 2},
		)
		if err != nil {
			t.Fatalf("DynamicIota with minted countAxis failed: %v", err)
		}
		oneScalar, err := mainFn.Constant([]int64{1})
		if err != nil {
			t.Fatalf("Constant failed: %v", err)
		}
		onesBroadcast, err := mainFn.BroadcastInDim(oneScalar, coordsShape, nil)
		if err != nil {
			t.Fatalf("BroadcastInDim to minted countAxis failed: %v", err)
		}
		sum1, err := mainFn.Add(coords, rowIds)
		if err != nil {
			t.Fatalf("Add(coords, rowIds) failed: %v", err)
		}
		sum2, err := mainFn.Add(sum1, onesBroadcast)
		if err != nil {
			t.Fatalf("Add(sum1, onesBroadcast) failed: %v", err)
		}

		if err := mainFn.Return([]compute.Value{coords, sum2}, nil); err != nil {
			t.Fatalf("Return failed: %v", err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer exec.Finalize()

		// Execute with 3 nonzeros, then 0 nonzeros, then 2 nonzeros on the same compiled executable.
		for _, tc := range []struct {
			rows       int
			flat       []float32
			wantCoords []int64
			wantSum    []int64
		}{
			{
				rows:       2,
				flat:       []float32{0, 4, 0, 2, 0, 3},
				wantCoords: []int64{0, 1, 1, 0, 1, 2},
				wantSum:    []int64{1, 2, 3, 2, 4, 5}, // coords + [[0,0],[1,1],[2,2]] + 1
			},
			{
				rows:       1,
				flat:       []float32{0, 0, 0},
				wantCoords: []int64{},
				wantSum:    []int64{},
			},
			{
				rows:       1,
				flat:       []float32{5, 0, 9},
				wantCoords: []int64{0, 0, 0, 2},
				wantSum:    []int64{1, 1, 2, 4},
			},
		} {
			inBuf, err := backend.BufferFromFlatData(0, tc.flat, shapes.Make(dtypes.Float32, tc.rows, 3))
			if err != nil {
				t.Fatalf("BufferFromFlatData failed: %v", err)
			}
			outs, err := exec.Execute([]compute.Buffer{inBuf}, nil, 0)
			inBuf.Finalize()
			if err != nil {
				t.Fatalf("Execute failed: %v", err)
			}
			gotCoords := make([]int64, len(tc.wantCoords))
			if err := outs[0].ToFlatData(gotCoords); err != nil {
				t.Fatalf("ToFlatData coords failed: %v", err)
			}
			gotSum := make([]int64, len(tc.wantSum))
			if err := outs[1].ToFlatData(gotSum); err != nil {
				t.Fatalf("ToFlatData sum failed: %v", err)
			}
			outs[0].Finalize()
			outs[1].Finalize()
			if !slices.Equal(gotCoords, tc.wantCoords) {
				t.Fatalf("coords = %v, want %v", gotCoords, tc.wantCoords)
			}
			if !slices.Equal(gotSum, tc.wantSum) {
				t.Fatalf("sum = %v, want %v", gotSum, tc.wantSum)
			}
		}
	})

	t.Run("NonZeroAllDTypesAndScalarZero", func(t *testing.T) {
		runNZ1D := func(t *testing.T, dtype dtypes.DType, flat any) []int64 {
			t.Helper()
			if !backend.Capabilities().DTypes[dtype] {
				t.Skipf("dtype %s not supported", dtype)
			}
			b := backend.Builder("nz_" + dtype.String())
			fn := b.Main()
			p, err := fn.Parameter("x", shapes.Make(dtype, 3), nil)
			if err != nil {
				t.Fatalf("Parameter(%s): %v", dtype, err)
			}
			nz, err := fn.NonZero(p, compute.NonZeroConfig{})
			if err != nil {
				t.Fatalf("NonZero(%s): %v", dtype, err)
			}
			if err := fn.Return([]compute.Value{nz}, nil); err != nil {
				t.Fatalf("Return(%s): %v", dtype, err)
			}
			ex, err := b.Compile()
			if err != nil {
				t.Fatalf("Compile(%s): %v", dtype, err)
			}
			defer ex.Finalize()
			inBuf, err := backend.BufferFromFlatData(0, flat, shapes.Make(dtype, 3))
			if err != nil {
				t.Fatalf("BufferFromFlatData(%s): %v", dtype, err)
			}
			defer inBuf.Finalize()
			outs, err := ex.Execute([]compute.Buffer{inBuf}, nil, 0)
			if err != nil {
				t.Fatalf("Execute(%s): %v", dtype, err)
			}
			defer outs[0].Finalize()
			sh, _ := outs[0].Shape()
			got := make([]int64, sh.Size())
			if err := outs[0].ToFlatData(got); err != nil {
				t.Fatalf("ToFlatData(%s): %v", dtype, err)
			}
			return got
		}

		for _, tc := range []struct {
			dtype dtypes.DType
			flat  any
		}{
			{dtypes.Bool, []bool{false, true, true}},
			{dtypes.Int8, []int8{0, -2, 3}},
			{dtypes.Int16, []int16{0, -2, 3}},
			{dtypes.Int32, []int32{0, -2, 3}},
			{dtypes.Int64, []int64{0, -2, 3}},
			{dtypes.Uint8, []uint8{0, 2, 3}},
			{dtypes.Uint16, []uint16{0, 2, 3}},
			{dtypes.Uint32, []uint32{0, 2, 3}},
			{dtypes.Uint64, []uint64{0, 2, uint64(1) << 63}},
			{dtypes.Float16, []float16.Float16{float16.FromFloat32(0), float16.FromFloat32(-1.5), float16.FromFloat32(2)}},
			{dtypes.BFloat16, []bfloat16.BFloat16{bfloat16.FromFloat32(0), bfloat16.FromFloat32(-1.5), bfloat16.FromFloat32(2)}},
			{dtypes.Float32, []float32{0, -1.5, 2}},
			{dtypes.Float64, []float64{0, 1e-300, -2.5}},
		} {
			t.Run(tc.dtype.String(), func(t *testing.T) {
				got := runNZ1D(t, tc.dtype, tc.flat)
				if !slices.Equal(got, []int64{1, 2}) {
					t.Fatalf("NonZero(%s) = %v, want [1 2]", tc.dtype, got)
				}
			})
		}

		// Scalar zero with both Transpose=false ([0, 0]) and Transpose=true ([0, 0]), and scalar nonzero with Transpose=true ([0, 1]).
		for _, transpose := range []bool{false, true} {
			b := backend.Builder("nz_scalar")
			fn := b.Main()
			p, err := fn.Parameter("x", shapes.Make(dtypes.Float32), nil)
			if err != nil {
				t.Fatal(err)
			}
			nz, err := fn.NonZero(p, compute.NonZeroConfig{Transpose: transpose})
			if err != nil {
				t.Fatal(err)
			}
			if err := fn.Return([]compute.Value{nz}, nil); err != nil {
				t.Fatal(err)
			}
			ex, err := b.Compile()
			if err != nil {
				t.Fatal(err)
			}
			defer ex.Finalize()

			for _, val := range []float32{0, 7} {
				inBuf, err := backend.BufferFromFlatData(0, []float32{val}, shapes.Make(dtypes.Float32))
				if err != nil {
					t.Fatal(err)
				}
				outs, err := ex.Execute([]compute.Buffer{inBuf}, nil, 0)
				inBuf.Finalize()
				if err != nil {
					t.Fatal(err)
				}
				sh, _ := outs[0].Shape()
				outs[0].Finalize()
				wantDims := []int{0, 0}
				if val != 0 {
					if transpose {
						wantDims = []int{0, 1}
					} else {
						wantDims = []int{1, 0}
					}
				}
				if !slices.Equal(sh.Dimensions, wantDims) {
					t.Fatalf("scalar(%v, transpose=%v) shape = %v, want %v", val, transpose, sh.Dimensions, wantDims)
				}
			}
		}
	})

	t.Run("NameAnonymousDynamicAxesInConvAndDot", func(t *testing.T) {
		builder := backend.Builder("test_conv_dot_named_anon")
		fn := builder.Main()

		inShape := shapes.MakeDynamic(dtypes.Float32, []int{1, shapes.DynamicDim, 1}, []string{"", shapes.AnonymousAxis, ""})
		x, err := fn.Parameter("x", inShape, nil)
		if err != nil {
			t.Fatal(err)
		}
		kernel, err := fn.Constant([]float32{1, 1}, 2, 1, 1)
		if err != nil {
			t.Fatal(err)
		}
		conv, err := fn.ConvGeneral(x, kernel, compute.ConvolveAxesConfig{
			InputBatch: 0, InputChannels: 2, InputSpatial: []int{1},
			KernelInputChannels: 1, KernelOutputChannels: 2, KernelSpatial: []int{0},
			OutputBatch: 0, OutputChannels: 2, OutputSpatial: []int{1},
		}, []int{2}, [][2]int{{0, 0}}, []int{1}, []int{1}, 1, 1)
		if err != nil {
			t.Fatal(err)
		}
		convShape, _ := fn.Shape(conv)
		if !strings.HasPrefix(convShape.AxisName(1), "#conv_") {
			t.Fatalf("expected #conv_* axis name on strided ConvGeneral output, got %q", convShape.AxisName(1))
		}
		// Binary op between conv and Neg(conv) requires AxisNameEqual to hold on the derived dynamic axis.
		negConv, err := fn.Neg(conv)
		if err != nil {
			t.Fatal(err)
		}
		zeroCheck, err := fn.Add(conv, negConv)
		if err != nil {
			t.Fatalf("Add(conv, Neg(conv)) failed: %v", err)
		}
		if err := fn.Return([]compute.Value{zeroCheck}, nil); err != nil {
			t.Fatal(err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Finalize()

		inBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5}, shapes.Make(dtypes.Float32, 1, 5, 1))
		if err != nil {
			t.Fatal(err)
		}
		defer inBuf.Finalize()
		outs, err := exec.Execute([]compute.Buffer{inBuf}, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer outs[0].Finalize()
		got := make([]float32, 2)
		if err := outs[0].ToFlatData(got); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, []float32{0, 0}) {
			t.Fatalf("got %v, want [0 0]", got)
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
