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

func TestDynamicOps(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer backend.Finalize()

	if !backend.Capabilities().DynamicAxes {
		t.Fatalf("expected DynamicAxes capability to be true")
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

		shapeData := make([]int32, 2)
		if err := outputs[0].ToFlatData(shapeData); err != nil {
			t.Fatalf("ToFlatData failed: %v", err)
		}
		if !slices.Equal(shapeData, []int32{2, 4}) {
			t.Fatalf("expected shape [2, 4], got %v", shapeData)
		}

		d0Data := make([]int32, 1)
		if err := outputs[1].ToFlatData(d0Data); err != nil {
			t.Fatalf("ToFlatData failed: %v", err)
		}
		if d0Data[0] != 2 {
			t.Fatalf("expected dim0=2, got %d", d0Data[0])
		}

		d1Data := make([]int32, 1)
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
}

