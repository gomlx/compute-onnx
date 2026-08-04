// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"bytes"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/stretchr/testify/require"
)

func TestDynamicOps(t *testing.T) {
	backend, err := New("")
	require.NoError(t, err)
	defer backend.Finalize()

	require.True(t, backend.Capabilities().DynamicAxes)

	t.Run("DynamicShapeAndSize", func(t *testing.T) {
		builder := backend.Builder("test_dyn_shape")
		mainFn := builder.Main()

		paramShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 4}, []string{"batch", ""})
		x, err := mainFn.Parameter("x", paramShape, nil)
		require.NoError(t, err)

		shapeVal, err := mainFn.DynamicShape(x)
		require.NoError(t, err)

		dim0, err := mainFn.DynamicDimensionSize(x, 0)
		require.NoError(t, err)

		dim1, err := mainFn.DynamicDimensionSize(x, 1)
		require.NoError(t, err)

		err = mainFn.Return([]compute.Value{shapeVal, dim0, dim1}, nil)
		require.NoError(t, err)

		exec, err := builder.Compile()
		require.NoError(t, err)
		defer exec.Finalize()

		inBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6, 7, 8}, shapes.Make(dtypes.Float32, 2, 4))
		require.NoError(t, err)
		defer inBuf.Finalize()

		outputs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
		require.NoError(t, err)
		require.Len(t, outputs, 3)

		shapeData := make([]int32, 2)
		require.NoError(t, outputs[0].ToFlatData(shapeData))
		require.Equal(t, []int32{2, 4}, shapeData)

		d0Data := make([]int32, 1)
		require.NoError(t, outputs[1].ToFlatData(d0Data))
		require.Equal(t, int32(2), d0Data[0])

		d1Data := make([]int32, 1)
		require.NoError(t, outputs[2].ToFlatData(d1Data))
		require.Equal(t, int32(4), d1Data[0])
	})

	t.Run("DynamicReshape", func(t *testing.T) {
		builder := backend.Builder("test_dyn_reshape")
		mainFn := builder.Main()

		paramShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, shapes.DynamicDim}, []string{"batch", "seq"})
		x, err := mainFn.Parameter("x", paramShape, nil)
		require.NoError(t, err)

		batchSize, err := mainFn.DynamicDimensionSize(x, 0)
		require.NoError(t, err)

		seqLen, err := mainFn.DynamicDimensionSize(x, 1)
		require.NoError(t, err)

		numTokens, err := mainFn.Mul(batchSize, seqLen)
		require.NoError(t, err)

		reshaped, err := mainFn.DynamicReshape(x, compute.DynamicDimensionSpec{
			Name:  "num_tokens",
			Value: numTokens,
		})
		require.NoError(t, err)

		err = mainFn.Return([]compute.Value{reshaped}, nil)
		require.NoError(t, err)

		exec, err := builder.Compile()
		require.NoError(t, err)
		defer exec.Finalize()

		inBuf, err := backend.BufferFromFlatData(0, []float32{10, 20, 30, 40, 50, 60}, shapes.Make(dtypes.Float32, 2, 3))
		require.NoError(t, err)
		defer inBuf.Finalize()

		outputs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
		require.NoError(t, err)
		require.Len(t, outputs, 1)

		outData := make([]float32, 6)
		require.NoError(t, outputs[0].ToFlatData(outData))
		require.Equal(t, []float32{10, 20, 30, 40, 50, 60}, outData)
		outShape, err := outputs[0].Shape()
		require.NoError(t, err)
		require.Equal(t, []int{6}, outShape.Dimensions)
	})

	t.Run("NodeDeduplicationAndConstantCaching", func(t *testing.T) {
		builder := backend.Builder("test_dedup")
		fn := builder.Main().(*Function)

		c1, err := fn.Constant([]int64{1, 2}, 2)
		require.NoError(t, err)
		c2, err := fn.Constant([]int64{1, 2}, 2)
		require.NoError(t, err)

		require.Same(t, c1, c2)

		p, err := fn.Parameter("x", shapes.Make(dtypes.Float32, 2, 2), nil)
		require.NoError(t, err)

		add1, err := fn.Add(p, p)
		require.NoError(t, err)
		add2, err := fn.Add(p, p)
		require.NoError(t, err)

		require.Same(t, add1, add2)
	})

	t.Run("SaveAndLoadDynamicModel", func(t *testing.T) {
		onBackend := backend.(*Backend)
		onBackend.SetKeepModelProto(true)
		defer onBackend.SetKeepModelProto(false)

		builder := backend.Builder("test_save_dyn")
		mainFn := builder.Main()

		paramShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 3}, []string{"batch", ""})
		p, err := mainFn.Parameter("x", paramShape, nil)
		require.NoError(t, err)

		doubleP, err := mainFn.Add(p, p)
		require.NoError(t, err)

		err = mainFn.Return([]compute.Value{doubleP}, nil)
		require.NoError(t, err)

		exec, err := builder.Compile()
		require.NoError(t, err)
		defer exec.Finalize()

		var buf bytes.Buffer
		err = SaveModel(backend, exec, &buf, []string{"input_x"}, []string{"output_y"})
		require.NoError(t, err)

		loadedExec, err := LoadModel(backend, &buf)
		require.NoError(t, err)
		defer loadedExec.Finalize()

		inBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, shapes.Make(dtypes.Float32, 2, 3))
		require.NoError(t, err)
		defer inBuf.Finalize()

		outBufs, err := loadedExec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
		require.NoError(t, err)
		require.Len(t, outBufs, 1)

		res := make([]float32, 6)
		require.NoError(t, outBufs[0].ToFlatData(res))
		require.Equal(t, []float32{2, 4, 6, 8, 10, 12}, res)
	})
}
