// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"math"
	"sync"
	"testing"

	"github.com/gomlx/compute"
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

	builder := backend.Builder("seq_deferred")
	mainFn := builder.Main()
	inputShape := shapes.Make(dtypes.Float32, inputSize)
	inVal, err := mainFn.Parameter("input", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter failed: %v", err)
	}
	twoConst, err := mainFn.Constant([]float32{2.0})
	if err != nil {
		t.Fatalf("Constant failed: %v", err)
	}
	outVal, err := mainFn.Mul(inVal, twoConst)
	if err != nil {
		t.Fatalf("Mul failed: %v", err)
	}
	if err := mainFn.Return([]compute.Value{outVal}, nil); err != nil {
		t.Fatalf("Return failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec.Finalize()

	outputBuffers := make([]compute.Buffer, numExecutions)
	expectedResults := make([][]float32, numExecutions)

	for i := 0; i < numExecutions; i++ {
		inputData := make([]float32, inputSize)
		expected := make([]float32, inputSize)
		for j := 0; j < inputSize; j++ {
			inputData[j] = float32(i*inputSize + j + 1)
			expected[j] = inputData[j] * 2.0
		}
		expectedResults[i] = expected

		inBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
		if err != nil {
			t.Fatalf("BufferFromFlatData failed: %v", err)
		}

		outputs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
		_ = inBuf.Finalize()
		if err != nil {
			t.Fatalf("Execution %d failed: %v", i, err)
		}
		outputBuffers[i] = outputs[0]
	}

	// Materialize and verify all output buffers
	for i, buf := range outputBuffers {
		result := make([]float32, inputSize)
		if err := buf.ToFlatData(result); err != nil {
			t.Fatalf("ToFlatData %d failed: %v", i, err)
		}
		for j := 0; j < inputSize; j++ {
			if math.Abs(float64(result[j]-expectedResults[i][j])) > 1e-5 {
				t.Errorf("Execution %d, element %d: expected %f, got %f",
					i, j, expectedResults[i][j], result[j])
			}
		}
		_ = buf.Finalize()
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

	builder := backend.Builder("concurrent_exec")
	mainFn := builder.Main()
	inputShape := shapes.Make(dtypes.Float32, inputSize)
	inVal, err := mainFn.Parameter("input", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter failed: %v", err)
	}
	twoConst, err := mainFn.Constant([]float32{2.0})
	if err != nil {
		t.Fatalf("Constant failed: %v", err)
	}
	outVal, err := mainFn.Mul(inVal, twoConst)
	if err != nil {
		t.Fatalf("Mul failed: %v", err)
	}
	if err := mainFn.Return([]compute.Value{outVal}, nil); err != nil {
		t.Fatalf("Return failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec.Finalize()

	type execResult struct {
		workerID int
		execID   int
		buf      compute.Buffer
		expected []float32
	}

	var mu sync.Mutex
	var allResults []execResult
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for e := 0; e < numExecutionsPerWorker; e++ {
				inputData := make([]float32, inputSize)
				expected := make([]float32, inputSize)
				for j := 0; j < inputSize; j++ {
					v := float32(workerID*1000 + e*100 + j + 1)
					inputData[j] = v
					expected[j] = v * 2.0
				}

				inBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
				if err != nil {
					t.Errorf("Worker %d, execution %d buffer creation failed: %v", workerID, e, err)
					return
				}

				outputs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
				_ = inBuf.Finalize()
				if err != nil {
					t.Errorf("Worker %d, execution %d failed: %v", workerID, e, err)
					return
				}

				mu.Lock()
				allResults = append(allResults, execResult{
					workerID: workerID,
					execID:   e,
					buf:      outputs[0],
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
		if err := r.buf.ToFlatData(result); err != nil {
			t.Fatalf("Worker %d, exec %d ToFlatData failed: %v", r.workerID, r.execID, err)
		}
		for j := 0; j < inputSize; j++ {
			if math.Abs(float64(result[j]-r.expected[j])) > 1e-5 {
				t.Errorf("Worker %d, exec %d, element %d: expected %f, got %f",
					r.workerID, r.execID, j, r.expected[j], result[j])
			}
		}
		_ = r.buf.Finalize()
	}

	t.Logf("Successfully verified %d results from %d workers", len(allResults), numWorkers)
}

// TestMultiInputHostTensorsCUDA tests execution on CUDA backend when passing multiple host-allocated parameters.
func TestMultiInputHostTensorsCUDA(t *testing.T) {
	if backend == nil {
		t.Skip("Backend not available")
	}

	builder := backend.Builder("multi_input")
	mainFn := builder.Main()
	sh := shapes.Make(dtypes.Float32, 4)
	a, err := mainFn.Parameter("a", sh, nil)
	if err != nil {
		t.Fatalf("Param a failed: %v", err)
	}
	b, err := mainFn.Parameter("b", sh, nil)
	if err != nil {
		t.Fatalf("Param b failed: %v", err)
	}
	c, err := mainFn.Parameter("c", sh, nil)
	if err != nil {
		t.Fatalf("Param c failed: %v", err)
	}

	sum, err := mainFn.Add(a, b)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	prod, err := mainFn.Mul(sum, c)
	if err != nil {
		t.Fatalf("Mul failed: %v", err)
	}
	if err := mainFn.Return([]compute.Value{prod}, nil); err != nil {
		t.Fatalf("Return failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec.Finalize()

	aData := []float32{1.0, 2.0, 3.0, 4.0}
	bData := []float32{10.0, 20.0, 30.0, 40.0}
	cData := []float32{2.0, 2.0, 2.0, 2.0}

	aBuf, _ := backend.BufferFromFlatData(0, aData, sh)
	bBuf, _ := backend.BufferFromFlatData(0, bData, sh)
	cBuf, _ := backend.BufferFromFlatData(0, cData, sh)
	defer aBuf.Finalize()
	defer bBuf.Finalize()
	defer cBuf.Finalize()

	outputs, err := exec.Execute([]compute.Buffer{aBuf, bBuf, cBuf}, []bool{false}, 0)
	if err != nil {
		t.Fatalf("Execution with multiple host inputs failed: %v", err)
	}
	defer outputs[0].Finalize()

	result := make([]float32, 4)
	if err := outputs[0].ToFlatData(result); err != nil {
		t.Fatalf("ToFlatData failed: %v", err)
	}
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

	builder := backend.Builder("gather_emb")
	mainFn := builder.Main()
	embShape := shapes.Make(dtypes.Float32, 4, 3)
	idxShape := shapes.Make(dtypes.Int64, 2, 2, 1)

	embTable, err := mainFn.Parameter("emb", embShape, nil)
	if err != nil {
		t.Fatalf("Param emb failed: %v", err)
	}
	indices, err := mainFn.Parameter("idx", idxShape, nil)
	if err != nil {
		t.Fatalf("Param idx failed: %v", err)
	}

	gathered, err := mainFn.Gather(embTable, indices, 2, []int{2}, []int{0}, []int{0}, []int{1, 3}, false)
	if err != nil {
		t.Fatalf("Gather failed: %v", err)
	}
	if err := mainFn.Return([]compute.Value{gathered}, nil); err != nil {
		t.Fatalf("Return failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec.Finalize()

	embedData := []float32{
		0.0, 0.1, 0.2, // idx 0
		1.0, 1.1, 1.2, // idx 1
		2.0, 2.1, 2.2, // idx 2
		3.0, 3.1, 3.2, // idx 3
	}
	indicesData := []int64{3, 1, 0, 2}

	embBuf, _ := backend.BufferFromFlatData(0, embedData, embShape)
	idxBuf, _ := backend.BufferFromFlatData(0, indicesData, idxShape)
	defer embBuf.Finalize()
	defer idxBuf.Finalize()

	outputs, err := exec.Execute([]compute.Buffer{embBuf, idxBuf}, []bool{false}, 0)
	if err != nil {
		t.Fatalf("Gather execution on CUDA failed: %v", err)
	}
	defer outputs[0].Finalize()

	result := make([]float32, 2*2*3)
	if err := outputs[0].ToFlatData(result); err != nil {
		t.Fatalf("ToFlatData failed: %v", err)
	}
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
