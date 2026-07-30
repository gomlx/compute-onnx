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

func TestSaveModel(t *testing.T) {
	backend, err := New("")
	require.NoError(t, err)
	defer backend.Finalize()

	onBackend, ok := backend.(*Backend)
	require.True(t, ok)

	// Step 1: Verify SaveModel returns error when SetKeepModelProto is false (default)
	builder := backend.Builder("test_save_disabled")
	fn := builder.Main()
	x, err := fn.Parameter("x", shapes.Make(dtypes.Float32, 2, 3), nil)
	require.NoError(t, err)
	sum, err := fn.Add(x, x)
	require.NoError(t, err)
	err = fn.Return([]compute.Value{sum}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)
	defer exec.Finalize()

	var buf bytes.Buffer
	err = SaveModel(backend, exec, &buf, nil, nil)
	require.Error(t, err)
	require.Contains(t, err.Error(), "backend.SetKeepModelProto(true) must be called")

	// Step 2: Enable SetKeepModelProto and save model with default input/output names
	onBackend.SetKeepModelProto(true)
	require.True(t, onBackend.KeepModelProto())

	builder2 := backend.Builder("test_save_enabled")
	fn2 := builder2.Main()
	p1, err := fn2.Parameter("in_a", shapes.Make(dtypes.Float32, 2, 2), nil)
	require.NoError(t, err)
	p2, err := fn2.Parameter("in_b", shapes.Make(dtypes.Float32, 2, 2), nil)
	require.NoError(t, err)
	sum2, err := fn2.Add(p1, p2)
	require.NoError(t, err)
	err = fn2.Return([]compute.Value{sum2}, nil)
	require.NoError(t, err)

	exec2, err := builder2.Compile()
	require.NoError(t, err)
	defer exec2.Finalize()

	buf.Reset()
	err = SaveModel(backend, exec2, &buf, nil, nil)
	require.NoError(t, err)
	require.NotEmpty(t, buf.Bytes())

	// Step 3: Save model with custom input and output names
	var bufCustom bytes.Buffer
	customInputs := []string{"custom_x", "custom_y"}
	customOutputs := []string{"custom_sum"}
	err = SaveModel(backend, exec2, &bufCustom, customInputs, customOutputs)
	require.NoError(t, err)
	require.NotEmpty(t, bufCustom.Bytes())

	// Verify the saved model bytes can be loaded into an ONNX Runtime session
	onExec2, ok := exec2.(*Executable)
	require.True(t, ok)
	require.NotNil(t, onExec2.ModelProto())
	require.Equal(t, "custom_x", onExec2.ModelProto().Graph.Input[0].Name)
	require.Equal(t, "custom_y", onExec2.ModelProto().Graph.Input[1].Name)
	require.Equal(t, "custom_sum", onExec2.ModelProto().Graph.Output[0].Name)
}

func TestLoadModel(t *testing.T) {
	backend, err := New("")
	require.NoError(t, err)
	defer backend.Finalize()

	onBackend, ok := backend.(*Backend)
	require.True(t, ok)
	onBackend.SetKeepModelProto(true)

	// Build a simple addition graph
	builder := backend.Builder("test_load")
	fn := builder.Main()
	p1, err := fn.Parameter("a", shapes.Make(dtypes.Float32, 3), nil)
	require.NoError(t, err)
	p2, err := fn.Parameter("b", shapes.Make(dtypes.Float32, 3), nil)
	require.NoError(t, err)
	out, err := fn.Add(p1, p2)
	require.NoError(t, err)
	err = fn.Return([]compute.Value{out}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)
	defer exec.Finalize()

	// Save to buffer
	var buf bytes.Buffer
	err = SaveModel(backend, exec, &buf, []string{"in_1", "in_2"}, []string{"out_sum"})
	require.NoError(t, err)

	// Load model from buffer via LoadModel
	loadedExec, err := LoadModel(backend, &buf)
	require.NoError(t, err)
	defer loadedExec.Finalize()

	inputs, _ := loadedExec.Inputs()
	require.Equal(t, []string{"in_1", "in_2"}, inputs)

	// Execute loaded model and verify results
	buf1, err := backend.BufferFromFlatData(0, []float32{1.0, 2.0, 3.0}, shapes.Make(dtypes.Float32, 3))
	require.NoError(t, err)
	defer buf1.Finalize()

	buf2, err := backend.BufferFromFlatData(0, []float32{4.0, 5.0, 6.0}, shapes.Make(dtypes.Float32, 3))
	require.NoError(t, err)
	defer buf2.Finalize()

	outBuffers, err := loadedExec.Execute([]compute.Buffer{buf1, buf2}, []bool{false, false}, 0)
	require.NoError(t, err)
	require.Len(t, outBuffers, 1)
	defer outBuffers[0].Finalize()

	resData := make([]float32, 3)
	err = outBuffers[0].ToFlatData(resData)
	require.NoError(t, err)
	require.InDeltaSlice(t, []float32{5.0, 7.0, 9.0}, resData, 1e-5)
}

func TestEuclideanDistanceEndToEnd(t *testing.T) {
	// End-to-end example: Build Euclidean Distance graph sqrt(sum((p - q)^2)), save, load, and execute.
	backend, err := New("")
	require.NoError(t, err)
	defer backend.Finalize()

	onBackend, ok := backend.(*Backend)
	require.True(t, ok)
	onBackend.SetKeepModelProto(true)

	// 1. Build Euclidean Distance graph: dist = sqrt(ReduceSum((p - q)^2))
	builder := backend.Builder("euclidean_distance")
	fn := builder.Main()

	p, err := fn.Parameter("point_p", shapes.Make(dtypes.Float32, 3), nil)
	require.NoError(t, err)
	q, err := fn.Parameter("point_q", shapes.Make(dtypes.Float32, 3), nil)
	require.NoError(t, err)

	// diff = p - q
	diff, err := fn.Sub(p, q)
	require.NoError(t, err)

	// diff_sq = diff * diff
	diffSq, err := fn.Mul(diff, diff)
	require.NoError(t, err)

	// sum_sq = ReduceSum(diff_sq) over axis 0
	sumSq, err := fn.ReduceSum(diffSq, 0)
	require.NoError(t, err)

	// dist = Sqrt(sum_sq)
	dist, err := fn.Sqrt(sumSq)
	require.NoError(t, err)

	err = fn.Return([]compute.Value{dist}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)
	defer exec.Finalize()

	// 2. Save the ONNX model to memory buffer with custom parameter names
	var onnxModelBuf bytes.Buffer
	err = SaveModel(backend, exec, &onnxModelBuf, []string{"p", "q"}, []string{"distance"})
	require.NoError(t, err)
	require.NotEmpty(t, onnxModelBuf.Bytes())

	// 3. Load the saved ONNX model back via LoadModel
	loadedExec, err := LoadModel(backend, &onnxModelBuf)
	require.NoError(t, err)
	defer loadedExec.Finalize()

	// 4. Test case 1: 3D Euclidean distance between (0, 0, 0) and (3, 4, 0) => sqrt(3^2 + 4^2) = 5.0
	bufP1, err := backend.BufferFromFlatData(0, []float32{0.0, 0.0, 0.0}, shapes.Make(dtypes.Float32, 3))
	require.NoError(t, err)
	defer bufP1.Finalize()

	bufQ1, err := backend.BufferFromFlatData(0, []float32{3.0, 4.0, 0.0}, shapes.Make(dtypes.Float32, 3))
	require.NoError(t, err)
	defer bufQ1.Finalize()

	res1, err := loadedExec.Execute([]compute.Buffer{bufP1, bufQ1}, []bool{false, false}, 0)
	require.NoError(t, err)
	require.Len(t, res1, 1)
	defer res1[0].Finalize()

	dist1 := make([]float32, 1)
	err = res1[0].ToFlatData(dist1)
	require.NoError(t, err)
	require.InDelta(t, float32(5.0), dist1[0], 1e-4)

	// 5. Test case 2: 3D Euclidean distance between (1, 2, 3) and (4, 6, 3) => sqrt(3^2 + 4^2 + 0^2) = 5.0
	bufP2, err := backend.BufferFromFlatData(0, []float32{1.0, 2.0, 3.0}, shapes.Make(dtypes.Float32, 3))
	require.NoError(t, err)
	defer bufP2.Finalize()

	bufQ2, err := backend.BufferFromFlatData(0, []float32{4.0, 6.0, 3.0}, shapes.Make(dtypes.Float32, 3))
	require.NoError(t, err)
	defer bufQ2.Finalize()

	res2, err := loadedExec.Execute([]compute.Buffer{bufP2, bufQ2}, []bool{false, false}, 0)
	require.NoError(t, err)
	require.Len(t, res2, 1)
	defer res2[0].Finalize()

	dist2 := make([]float32, 1)
	err = res2[0].ToFlatData(dist2)
	require.NoError(t, err)
	require.InDelta(t, float32(5.0), dist2[0], 1e-4)
}
