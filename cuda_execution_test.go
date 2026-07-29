// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxruntime

import (
	"math"
	"sync"
	"testing"

	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

// TestSequentialDeferredMaterialization tests that multiple sequential executions
// can keep their output tensors alive and materialize them only after all executions finish.
// This verifies that output buffers don't share underlying memory that gets overwritten.
func TestSequentialDeferredMaterialization(t *testing.T) {
	if backend == nil {
		t.Skip("Backend not available")
	}

	numExecutions := 10
	inputSize := 8

	// Build a simple graph: output = input * 2
	exec, err := graph.NewExec(backend, func(input *graph.Node) *graph.Node {
		return graph.MulScalar(input, 2.0)
	})
	if err != nil {
		t.Fatalf("NewExec failed: %v", err)
	}

	// Execute N times with different inputs, keep all output tensors.
	outputTensors := make([]*tensors.Tensor, numExecutions)
	expectedResults := make([][]float32, numExecutions)

	for i := range numExecutions {
		inputData := make([]float32, inputSize)
		expected := make([]float32, inputSize)
		for j := range inputSize {
			inputData[j] = float32(i*inputSize + j + 1)
			expected[j] = inputData[j] * 2.0
		}
		expectedResults[i] = expected

		inputTensor := tensors.FromFlatDataAndDimensions(inputData, inputSize)
		outputs, err := exec.Call(inputTensor)
		if err != nil {
			t.Fatalf("Execution %d failed: %v", i, err)
		}
		outputTensors[i] = outputs[0]
		// Don't materialize yet — keep the tensor alive.
	}

	// Now materialize all outputs and verify correctness.
	for i, tensor := range outputTensors {
		result := make([]float32, inputSize)
		tensor.ConstFlatData(func(flat any) {
			copy(result, flat.([]float32))
		})
		for j := range inputSize {
			if math.Abs(float64(result[j]-expectedResults[i][j])) > 1e-5 {
				t.Errorf("Execution %d, element %d: expected %f, got %f",
					i, j, expectedResults[i][j], result[j])
			}
		}
		_ = tensor.FinalizeAll()
	}
	t.Logf("Successfully verified %d sequential results with deferred materialization", numExecutions)
}

// TestConcurrentExecution tests that multiple goroutines can execute the same
// executable concurrently and get correct, independent results.
func TestConcurrentExecution(t *testing.T) {
	if backend == nil {
		t.Skip("Backend not available")
	}

	numWorkers := 8
	numExecutionsPerWorker := 5
	inputSize := 8

	// Build a simple graph: output = input * 2
	exec, err := graph.NewExec(backend, func(input *graph.Node) *graph.Node {
		return graph.MulScalar(input, 2.0)
	})
	if err != nil {
		t.Fatalf("NewExec failed: %v", err)
	}

	type execResult struct {
		workerID int
		execID   int
		tensor   *tensors.Tensor
		expected []float32
	}

	var mu sync.Mutex
	var allResults []execResult
	var wg sync.WaitGroup

	for w := range numWorkers {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for e := range numExecutionsPerWorker {
				inputData := make([]float32, inputSize)
				expected := make([]float32, inputSize)
				for j := range inputSize {
					v := float32(workerID*1000 + e*100 + j + 1)
					inputData[j] = v
					expected[j] = v * 2.0
				}

				inputTensor := tensors.FromFlatDataAndDimensions(inputData, inputSize)
				outputs, err := exec.Call(inputTensor)
				if err != nil {
					t.Errorf("Worker %d, execution %d failed: %v", workerID, e, err)
					return
				}

				mu.Lock()
				allResults = append(allResults, execResult{
					workerID: workerID,
					execID:   e,
					tensor:   outputs[0],
					expected: expected,
				})
				mu.Unlock()
			}
		}(w)
	}

	wg.Wait()

	// Verify all results (deferred materialization).
	for _, r := range allResults {
		result := make([]float32, inputSize)
		r.tensor.ConstFlatData(func(flat any) {
			copy(result, flat.([]float32))
		})
		for j := range inputSize {
			if math.Abs(float64(result[j]-r.expected[j])) > 1e-5 {
				t.Errorf("Worker %d, exec %d, element %d: expected %f, got %f",
					r.workerID, r.execID, j, r.expected[j], result[j])
			}
		}
		_ = r.tensor.FinalizeAll()
	}

	t.Logf("Successfully verified %d results from %d workers", len(allResults), numWorkers)
}

// TestMultiInputHostTensorsCUDA tests execution on CUDA backend when passing multiple host-allocated parameters.
func TestMultiInputHostTensorsCUDA(t *testing.T) {
	if backend == nil {
		t.Skip("Backend not available")
	}

	exec, err := graph.NewExec(backend, func(a, b, c *graph.Node) *graph.Node {
		// Calculate (a + b) * c
		sum := graph.Add(a, b)
		return graph.Mul(sum, c)
	})
	if err != nil {
		t.Fatalf("NewExec failed: %v", err)
	}

	aData := []float32{1.0, 2.0, 3.0, 4.0}
	bData := []float32{10.0, 20.0, 30.0, 40.0}
	cData := []float32{2.0, 2.0, 2.0, 2.0}

	aTensor := tensors.FromFlatDataAndDimensions(aData, 4)
	bTensor := tensors.FromFlatDataAndDimensions(bData, 4)
	cTensor := tensors.FromFlatDataAndDimensions(cData, 4)

	outputs, err := exec.Call(aTensor, bTensor, cTensor)
	if err != nil {
		t.Fatalf("Execution with multiple host inputs failed: %v", err)
	}

	result := make([]float32, 4)
	outputs[0].ConstFlatData(func(flat any) {
		copy(result, flat.([]float32))
	})

	expected := []float32{22.0, 44.0, 66.0, 88.0}
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > 1e-5 {
			t.Errorf("Element %d: expected %f, got %f", i, expected[i], result[i])
		}
	}
}

// TestGatherEmbeddingCUDA tests embedding lookup via Gather on axis 0 with CUDA backend.
func TestGatherEmbeddingCUDA(t *testing.T) {
	if backend == nil {
		t.Skip("Backend not available")
	}

	exec, err := graph.NewExec(backend, func(embeddingTable, indices *graph.Node) *graph.Node {
		return graph.Gather(embeddingTable, indices)
	})
	if err != nil {
		t.Fatalf("NewExec failed: %v", err)
	}

	// Embedding table: shape (4, 3)
	embedData := []float32{
		0.0, 0.1, 0.2, // idx 0
		1.0, 1.1, 1.2, // idx 1
		2.0, 2.1, 2.2, // idx 2
		3.0, 3.1, 3.2, // idx 3
	}
	embedTensor := tensors.FromFlatDataAndDimensions(embedData, 4, 3)

	// Indices: shape (2, 2, 1)
	indicesData := []int64{3, 1, 0, 2}
	indicesTensor := tensors.FromFlatDataAndDimensions(indicesData, 2, 2, 1)

	outputs, err := exec.Call(embedTensor, indicesTensor)
	if err != nil {
		t.Fatalf("Gather execution on CUDA failed: %v", err)
	}

	result := make([]float32, 2*2*3)
	outputs[0].ConstFlatData(func(flat any) {
		copy(result, flat.([]float32))
	})

	expected := []float32{
		3.0, 3.1, 3.2, // idx 3
		1.0, 1.1, 1.2, // idx 1
		0.0, 0.1, 0.2, // idx 0
		2.0, 2.1, 2.2, // idx 2
	}

	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > 1e-5 {
			t.Errorf("Element %d: expected %f, got %f", i, expected[i], result[i])
		}
	}
}

// Ensure unused imports are accounted for.
var (
	_ = dtypes.Float32
	_ = shapes.Make
)
