// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !js

package ort

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	onnx "github.com/gomlx/compute-onnx/support/protos"
	"google.golang.org/protobuf/proto"
)

func createIdentityModel(size int64) []byte {
	model := &onnx.ModelProto{
		IrVersion: 9,
		Graph: &onnx.GraphProto{
			Name: "identity_graph",
			Node: []*onnx.NodeProto{
				{
					Input:  []string{"x"},
					Output: []string{"y"},
					OpType: "Identity",
				},
			},
			Input: []*onnx.ValueInfoProto{
				{
					Name: "x",
					Type: &onnx.TypeProto{
						Value: &onnx.TypeProto_TensorType{
							TensorType: &onnx.TypeProto_Tensor{
								ElemType: int32(onnx.TensorProto_FLOAT),
								Shape: &onnx.TensorShapeProto{
									Dim: []*onnx.TensorShapeProto_Dimension{
										{Value: &onnx.TensorShapeProto_Dimension_DimValue{DimValue: size}},
									},
								},
							},
						},
					},
				},
			},
			Output: []*onnx.ValueInfoProto{
				{
					Name: "y",
					Type: &onnx.TypeProto{
						Value: &onnx.TypeProto_TensorType{
							TensorType: &onnx.TypeProto_Tensor{
								ElemType: int32(onnx.TensorProto_FLOAT),
								Shape: &onnx.TensorShapeProto{
									Dim: []*onnx.TensorShapeProto_Dimension{
										{Value: &onnx.TensorShapeProto_Dimension_DimValue{DimValue: size}},
									},
								},
							},
						},
					},
				},
			},
		},
		OpsetImport: []*onnx.OperatorSetIdProto{
			{
				Domain:  "",
				Version: 21,
			},
		},
	}
	bytes, err := proto.Marshal(model)
	if err != nil {
		panic(err)
	}
	return bytes
}

func initTestEnvironment() error {
	path := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")
	if path == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return err
		}
		path = filepath.Join(home, ".local", "lib", "onnxruntime", "libonnxruntime.so")
	}
	SetSharedLibraryPath(path)
	return InitializeEnvironment()
}

func TestBasicInference(t *testing.T) {
	if err := initTestEnvironment(); err != nil {
		t.Fatalf("Failed to initialize environment: %v", err)
	}

	env, err := NewEnv("test_env")
	if err != nil {
		t.Fatalf("NewEnv failed: %v", err)
	}
	defer env.Destroy()

	modelBytes := createIdentityModel(10)
	session, err := NewSessionWithONNXData(env, modelBytes, nil)
	if err != nil {
		t.Fatalf("NewSession failed: %v", err)
	}
	defer session.Destroy()

	inputData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	inputTensor, err := NewTensor(NewShape(10), inputData)
	if err != nil {
		t.Fatalf("NewTensor failed: %v", err)
	}
	defer inputTensor.Destroy()

	outputTensor, err := NewEmptyTensor[float32](NewShape(10))
	if err != nil {
		t.Fatalf("NewEmptyTensor failed: %v", err)
	}
	defer outputTensor.Destroy()

	err = session.Run([]string{"x"}, []Value{inputTensor}, []string{"y"}, []Value{outputTensor})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	outputSlice := outputTensor.GetData()

	for i, v := range outputSlice {
		if v != inputData[i] {
			t.Errorf("Mismatch at index %d: expected %f, got %f", i, inputData[i], v)
		}
	}
}

func BenchmarkInference(b *testing.B) {
	if err := initTestEnvironment(); err != nil {
		b.Fatalf("Failed to initialize environment: %v", err)
	}

	env, err := NewEnv("bench_env")
	if err != nil {
		b.Fatalf("NewEnv failed: %v", err)
	}
	defer env.Destroy()

	sizes := []int64{1, 1024, 1048576}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			modelBytes := createIdentityModel(size)
			session, err := NewSessionWithONNXData(env, modelBytes, nil)
			if err != nil {
				b.Fatalf("NewSession failed: %v", err)
			}
			defer session.Destroy()

			inputData := make([]float32, size)
			inputTensor, err := NewTensor(NewShape(size), inputData)
			if err != nil {
				b.Fatalf("NewTensor failed: %v", err)
			}
			defer inputTensor.Destroy()

			outputTensor, err := NewEmptyTensor[float32](NewShape(size))
			if err != nil {
				b.Fatalf("NewEmptyTensor failed: %v", err)
			}
			defer outputTensor.Destroy()

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				err = session.Run([]string{"x"}, []Value{inputTensor}, []string{"y"}, []Value{outputTensor})
				if err != nil {
					b.Fatalf("Run failed: %v", err)
				}
			}
		})
	}
}
