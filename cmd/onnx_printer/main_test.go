// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	onnxbackend "github.com/gomlx/compute-onnx"
	"github.com/gomlx/compute-onnx/internal/protos"
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"google.golang.org/protobuf/proto"
)

func TestPrintModel(t *testing.T) {
	backend, err := onnxbackend.New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %+v", err)
	}
	defer backend.Finalize()
	backend.(*onnxbackend.Backend).SetKeepModelProto(true)

	builder := backend.Builder("test_printer")
	fn := builder.Main()

	// Parameter with static shape
	x, err := fn.Parameter("x", shapes.Make(dtypes.Float32, 2, 3), nil)
	if err != nil {
		t.Fatalf("Failed to create parameter x: %+v", err)
	}

	// Constant with 15 elements (> 10 max items)
	constData := make([]float32, 15)
	for i := range constData {
		constData[i] = float32(i + 1)
	}
	c, err := fn.Constant(constData, 15)
	if err != nil {
		t.Fatalf("Failed to create constant: %+v", err)
	}

	// Dynamic parameter
	dynParamShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 3}, []string{"batch", ""})
	dynParam, err := fn.Parameter("dyn_x", dynParamShape, nil)
	if err != nil {
		t.Fatalf("Failed to create dynamic parameter: %+v", err)
	}

	// Add node
	addNode, err := fn.Add(x, x)
	if err != nil {
		t.Fatalf("Failed to create Add node: %+v", err)
	}

	fn.Return([]compute.Value{addNode, c, dynParam}, nil)

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Failed to compile graph: %+v", err)
	}

	// Get saved ModelProto from compile
	savedModel := exec.(*onnxbackend.Executable).ModelProto()
	if savedModel == nil {
		t.Fatalf("Expected non-nil ModelProto from Executable")
	}

	// Test default maxItems (10)
	var buf bytes.Buffer
	PrintModel(&buf, savedModel, "test_model.onnx", 10, true)
	outStr := buf.String()

	t.Logf("Printed Model:\n%s", outStr)

	// Check output assertions
	if !strings.Contains(outStr, "ONNX Model: test_model.onnx") {
		t.Errorf("Expected model header, got:\n%s", outStr)
	}

	// Should contain shape strings formatted by shapes.Shape
	if !strings.Contains(outStr, "(Float32)[2, 3]") {
		t.Errorf("Expected shape (Float32)[2, 3] in output, got:\n%s", outStr)
	}
	if !strings.Contains(outStr, "(Float32)[batch=?, 3]") {
		t.Errorf("Expected dynamic shape (Float32)[batch=?, 3] in output, got:\n%s", outStr)
	}
	if !strings.Contains(outStr, "[0] node_2: Add(x:(Float32)[2, 3], x:(Float32)[2, 3]) : (Float32)[2, 3]") {
		t.Errorf("Expected single line op formatting in output, got:\n%s", outStr)
	}

	// Initializer constant with 15 elements truncated at 10 items
	if !strings.Contains(outStr, ", ...]") {
		t.Errorf("Expected truncated constant with '...', got:\n%s", outStr)
	}

	// Test custom maxItems (5)
	buf.Reset()
	PrintModel(&buf, savedModel, "test_model.onnx", 5, false)
	outStr5 := buf.String()

	if !strings.Contains(outStr5, "[1, 2, 3, 4, 5, ...]") {
		t.Errorf("Expected 5 items truncated constant '[1, 2, 3, 4, 5, ...]', got:\n%s", outStr5)
	}
}

func TestFileAndStdinInput(t *testing.T) {
	// Build a simple ModelProto and serialize it to file
	model := &protos.ModelProto{
		IrVersion: 9,
		Graph: &protos.GraphProto{
			Name: "simple_graph",
			Input: []*protos.ValueInfoProto{
				{
					Name: "input_a",
					Type: &protos.TypeProto{
						Value: &protos.TypeProto_TensorType{
							TensorType: &protos.TypeProto_Tensor{
								ElemType: int32(protos.TensorProto_FLOAT),
								Shape: &protos.TensorShapeProto{
									Dim: []*protos.TensorShapeProto_Dimension{
										{Value: &protos.TensorShapeProto_Dimension_DimValue{DimValue: 4}},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	data, err := proto.Marshal(model)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	tempDir := t.TempDir()
	filePath := filepath.Join(tempDir, "test.onnx")
	if err := os.WriteFile(filePath, data, 0644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	var buf bytes.Buffer
	PrintModel(&buf, model, filePath, 10, true)
	outStr := buf.String()

	if !strings.Contains(outStr, "input_a: (Float32)[4]") {
		t.Errorf("Expected input shape (Float32)[4], got:\n%s", outStr)
	}
}
