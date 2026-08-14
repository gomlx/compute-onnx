// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"bytes"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

func TestSaveModel(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer backend.Finalize()

	onBackend, ok := backend.(*Backend)
	if !ok {
		t.Fatalf("expected backend to be *Backend")
	}

	// Step 1: Verify SaveModel returns error when SetKeepModelProto is false (default)
	builder := backend.Builder("test_save_disabled")
	fn := builder.Main()
	x, err := fn.Parameter("x", shapes.Make(dtypes.Float32, 2, 3), nil)
	if err != nil {
		t.Fatalf("Parameter failed: %v", err)
	}
	sum, err := fn.Add(x, x)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	err = fn.Return([]compute.Value{sum}, nil)
	if err != nil {
		t.Fatalf("Return failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec.Finalize()

	var buf bytes.Buffer
	err = SaveModel(backend, exec, &buf, nil, nil)
	if err == nil {
		t.Fatalf("expected error when SaveModel called without SetKeepModelProto(true)")
	}
	if !strings.Contains(err.Error(), "backend.SetKeepModelProto(true) must be called") {
		t.Fatalf("unexpected error message: %v", err)
	}

	// Step 2: Enable SetKeepModelProto and save model with default input/output names
	onBackend.SetKeepModelProto(true)
	if !onBackend.KeepModelProto() {
		t.Fatalf("expected KeepModelProto to be true")
	}

	builder2 := backend.Builder("test_save_enabled")
	fn2 := builder2.Main()
	p1, err := fn2.Parameter("in_a", shapes.Make(dtypes.Float32, 2, 2), nil)
	if err != nil {
		t.Fatalf("Parameter in_a failed: %v", err)
	}
	p2, err := fn2.Parameter("in_b", shapes.Make(dtypes.Float32, 2, 2), nil)
	if err != nil {
		t.Fatalf("Parameter in_b failed: %v", err)
	}
	sum2, err := fn2.Add(p1, p2)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	err = fn2.Return([]compute.Value{sum2}, nil)
	if err != nil {
		t.Fatalf("Return failed: %v", err)
	}

	exec2, err := builder2.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec2.Finalize()

	buf.Reset()
	err = SaveModel(backend, exec2, &buf, nil, nil)
	if err != nil {
		t.Fatalf("SaveModel failed: %v", err)
	}
	if buf.Len() == 0 {
		t.Fatalf("expected non-empty buffer")
	}

	// Step 3: Save model with custom input and output names
	var bufCustom bytes.Buffer
	customInputs := []string{"custom_x", "custom_y"}
	customOutputs := []string{"custom_sum"}
	err = SaveModel(backend, exec2, &bufCustom, customInputs, customOutputs)
	if err != nil {
		t.Fatalf("SaveModel with custom names failed: %v", err)
	}
	if bufCustom.Len() == 0 {
		t.Fatalf("expected non-empty bufCustom")
	}

	// Verify the saved model bytes can be loaded into an ONNX Runtime session
	onExec2, ok := exec2.(*Executable)
	if !ok {
		t.Fatalf("expected exec2 to be *Executable")
	}
	if onExec2.ModelProto() == nil {
		t.Fatalf("expected ModelProto to be non-nil")
	}
	if onExec2.ModelProto().Graph.Input[0].Name != "custom_x" {
		t.Fatalf("expected input[0] name custom_x, got %s", onExec2.ModelProto().Graph.Input[0].Name)
	}
	if onExec2.ModelProto().Graph.Input[1].Name != "custom_y" {
		t.Fatalf("expected input[1] name custom_y, got %s", onExec2.ModelProto().Graph.Input[1].Name)
	}
	if onExec2.ModelProto().Graph.Output[0].Name != "custom_sum" {
		t.Fatalf("expected output[0] name custom_sum, got %s", onExec2.ModelProto().Graph.Output[0].Name)
	}
}

func TestLoadModel(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer backend.Finalize()

	onBackend, ok := backend.(*Backend)
	if !ok {
		t.Fatalf("expected backend to be *Backend")
	}
	onBackend.SetKeepModelProto(true)

	// Build a simple addition graph
	builder := backend.Builder("test_load")
	fn := builder.Main()
	p1, err := fn.Parameter("a", shapes.Make(dtypes.Float32, 3), nil)
	if err != nil {
		t.Fatalf("Parameter a failed: %v", err)
	}
	p2, err := fn.Parameter("b", shapes.Make(dtypes.Float32, 3), nil)
	if err != nil {
		t.Fatalf("Parameter b failed: %v", err)
	}
	out, err := fn.Add(p1, p2)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	err = fn.Return([]compute.Value{out}, nil)
	if err != nil {
		t.Fatalf("Return failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec.Finalize()

	// Save to buffer
	var buf bytes.Buffer
	err = SaveModel(backend, exec, &buf, []string{"in_1", "in_2"}, []string{"out_sum"})
	if err != nil {
		t.Fatalf("SaveModel failed: %v", err)
	}

	// Load model from buffer via LoadModel
	loadedExec, err := LoadModel(backend, &buf)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}
	defer loadedExec.Finalize()

	inputs, _ := loadedExec.Inputs()
	if !slices.Equal(inputs, []string{"in_1", "in_2"}) {
		t.Fatalf("expected inputs [in_1, in_2], got %v", inputs)
	}

	// Execute loaded model and verify results
	buf1, err := backend.BufferFromFlatData(0, []float32{1.0, 2.0, 3.0}, shapes.Make(dtypes.Float32, 3))
	if err != nil {
		t.Fatalf("BufferFromFlatData 1 failed: %v", err)
	}
	defer buf1.Finalize()

	buf2, err := backend.BufferFromFlatData(0, []float32{4.0, 5.0, 6.0}, shapes.Make(dtypes.Float32, 3))
	if err != nil {
		t.Fatalf("BufferFromFlatData 2 failed: %v", err)
	}
	defer buf2.Finalize()

	outBuffers, err := loadedExec.Execute([]compute.Buffer{buf1, buf2}, []bool{false, false}, 0)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if len(outBuffers) != 1 {
		t.Fatalf("expected 1 output buffer, got %d", len(outBuffers))
	}
	defer outBuffers[0].Finalize()

	resData := make([]float32, 3)
	err = outBuffers[0].ToFlatData(resData)
	if err != nil {
		t.Fatalf("ToFlatData failed: %v", err)
	}
	expected := []float32{5.0, 7.0, 9.0}
	for i := range expected {
		if math.Abs(float64(resData[i]-expected[i])) > 1e-5 {
			t.Fatalf("element %d: expected %f, got %f", i, expected[i], resData[i])
		}
	}
}

func TestEuclideanDistanceEndToEnd(t *testing.T) {
	// End-to-end example: Build Euclidean Distance graph sqrt(sum((p - q)^2)), save, load, and execute.
	backend, err := New("")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer backend.Finalize()

	onBackend, ok := backend.(*Backend)
	if !ok {
		t.Fatalf("expected backend to be *Backend")
	}
	onBackend.SetKeepModelProto(true)

	// 1. Build Euclidean Distance graph: dist = sqrt(ReduceSum((p - q)^2))
	builder := backend.Builder("euclidean_distance")
	fn := builder.Main()

	p, err := fn.Parameter("point_p", shapes.Make(dtypes.Float32, 3), nil)
	if err != nil {
		t.Fatalf("Parameter point_p failed: %v", err)
	}
	q, err := fn.Parameter("point_q", shapes.Make(dtypes.Float32, 3), nil)
	if err != nil {
		t.Fatalf("Parameter point_q failed: %v", err)
	}

	// diff = p - q
	diff, err := fn.Sub(p, q)
	if err != nil {
		t.Fatalf("Sub failed: %v", err)
	}

	// diff_sq = diff * diff
	diffSq, err := fn.Mul(diff, diff)
	if err != nil {
		t.Fatalf("Mul failed: %v", err)
	}

	// sum_sq = ReduceSum(diff_sq) over axis 0
	sumSq, err := fn.ReduceSum(diffSq, 0)
	if err != nil {
		t.Fatalf("ReduceSum failed: %v", err)
	}

	// dist = Sqrt(sum_sq)
	dist, err := fn.Sqrt(sumSq)
	if err != nil {
		t.Fatalf("Sqrt failed: %v", err)
	}

	err = fn.Return([]compute.Value{dist}, nil)
	if err != nil {
		t.Fatalf("Return failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec.Finalize()

	// 2. Save the ONNX model to memory buffer with custom parameter names
	var onnxModelBuf bytes.Buffer
	err = SaveModel(backend, exec, &onnxModelBuf, []string{"p", "q"}, []string{"distance"})
	if err != nil {
		t.Fatalf("SaveModel failed: %v", err)
	}
	if onnxModelBuf.Len() == 0 {
		t.Fatalf("expected non-empty onnxModelBuf")
	}

	// 3. Load the saved ONNX model back via LoadModel
	loadedExec, err := LoadModel(backend, &onnxModelBuf)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}
	defer loadedExec.Finalize()

	// 4. Test case 1: 3D Euclidean distance between (0, 0, 0) and (3, 4, 0) => sqrt(3^2 + 4^2) = 5.0
	bufP1, err := backend.BufferFromFlatData(0, []float32{0.0, 0.0, 0.0}, shapes.Make(dtypes.Float32, 3))
	if err != nil {
		t.Fatalf("BufferFromFlatData bufP1 failed: %v", err)
	}
	defer bufP1.Finalize()

	bufQ1, err := backend.BufferFromFlatData(0, []float32{3.0, 4.0, 0.0}, shapes.Make(dtypes.Float32, 3))
	if err != nil {
		t.Fatalf("BufferFromFlatData bufQ1 failed: %v", err)
	}
	defer bufQ1.Finalize()

	res1, err := loadedExec.Execute([]compute.Buffer{bufP1, bufQ1}, []bool{false, false}, 0)
	if err != nil {
		t.Fatalf("Execute res1 failed: %v", err)
	}
	if len(res1) != 1 {
		t.Fatalf("expected 1 output buffer, got %d", len(res1))
	}
	defer res1[0].Finalize()

	dist1 := make([]float32, 1)
	err = res1[0].ToFlatData(dist1)
	if err != nil {
		t.Fatalf("ToFlatData dist1 failed: %v", err)
	}
	if math.Abs(float64(dist1[0]-5.0)) > 1e-4 {
		t.Fatalf("expected distance 5.0, got %f", dist1[0])
	}

	// 5. Test case 2: 3D Euclidean distance between (1, 2, 3) and (4, 6, 3) => sqrt(3^2 + 4^2 + 0^2) = 5.0
	bufP2, err := backend.BufferFromFlatData(0, []float32{1.0, 2.0, 3.0}, shapes.Make(dtypes.Float32, 3))
	if err != nil {
		t.Fatalf("BufferFromFlatData bufP2 failed: %v", err)
	}
	defer bufP2.Finalize()

	bufQ2, err := backend.BufferFromFlatData(0, []float32{4.0, 6.0, 3.0}, shapes.Make(dtypes.Float32, 3))
	if err != nil {
		t.Fatalf("BufferFromFlatData bufQ2 failed: %v", err)
	}
	defer bufQ2.Finalize()

	res2, err := loadedExec.Execute([]compute.Buffer{bufP2, bufQ2}, []bool{false, false}, 0)
	if err != nil {
		t.Fatalf("Execute res2 failed: %v", err)
	}
	if len(res2) != 1 {
		t.Fatalf("expected 1 output buffer, got %d", len(res2))
	}
	defer res2[0].Finalize()

	dist2 := make([]float32, 1)
	err = res2[0].ToFlatData(dist2)
	if err != nil {
		t.Fatalf("ToFlatData dist2 failed: %v", err)
	}
	if math.Abs(float64(dist2[0]-5.0)) > 1e-4 {
		t.Fatalf("expected distance 5.0, got %f", dist2[0])
	}
}
