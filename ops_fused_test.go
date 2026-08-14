// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func TestFusedDenseActivationsAndLayouts(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %+v", err)
	}
	defer b.Finalize()

	x := [][]float32{{1, -2, 3}}
	// Weight in DenseLayoutOutputsInput shape [out_features, in_features] = [2, 3]
	w := [][]float32{{1, 0, 1}, {0, 1, 0}}
	bias := []float32{1, 2}

	t.Run("DenseLayoutOutputsInput", func(t *testing.T) {
		got, err := testutil.Exec1(b, []any{x, w, bias}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedDense(params[0], params[1], params[2], compute.DenseConfig{
				Activation:   compute.ActivationNone,
				WeightLayout: compute.DenseLayoutOutputsInput,
			})
		})
		if err != nil {
			t.Fatalf("FusedDense failed: %+v", err)
		}
		// x @ w^T + bias = [[1, -2, 3]] @ [[1, 0], [0, 1], [1, 0]] + [[1, 2]] = [[4, -2]] + [[1, 2]] = [[5, 0]]
		want := [][]float32{{5, 0}}
		if ok, diff := testutil.IsInDelta(want, got, 1e-4); !ok {
			t.Errorf("Mismatch:\n%s", diff)
		}
	})

	t.Run("ActivationSilu", func(t *testing.T) {
		xVal := []float32{-2.0, 0.0, 2.0}
		wVal := [][]float32{{1.0}, {1.0}, {1.0}}
		got, err := testutil.Exec1(b, []any{xVal, wVal}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedDense(params[0], params[1], nil, compute.DenseConfig{
				Activation:   compute.ActivationSilu,
				WeightLayout: compute.DenseLayoutInputOutputs,
			})
		})
		if err != nil {
			t.Fatalf("FusedDense Silu failed: %+v", err)
		}
		// sum = 0. silu(0) = 0 * sig(0) = 0
		want := []float32{0.0}
		if ok, diff := testutil.IsInDelta(want, got, 1e-4); !ok {
			t.Errorf("Mismatch:\n%s", diff)
		}
	})

	t.Run("ActivationTanh", func(t *testing.T) {
		xVal := []float32{0.0}
		wVal := [][]float32{{1.0}}
		got, err := testutil.Exec1(b, []any{xVal, wVal}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedDense(params[0], params[1], nil, compute.DenseConfig{
				Activation:   compute.ActivationTanh,
				WeightLayout: compute.DenseLayoutInputOutputs,
			})
		})
		if err != nil {
			t.Fatalf("FusedDense Tanh failed: %+v", err)
		}
		want := []float32{0.0}
		if ok, diff := testutil.IsInDelta(want, got, 1e-4); !ok {
			t.Errorf("Mismatch:\n%s", diff)
		}
	})
}

func TestFusedQuantizedDense(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %+v", err)
	}
	defer b.Finalize()

	x := [][]float32{{1.0, 2.0}}
	// Integer weights [2, 2]
	weightsInt := [][]int32{{10, 20}, {30, 40}}
	scale := []float32{0.1, 0.1}
	zeroPoint := []int32{0, 0}

	got, err := testutil.Exec1(b, []any{x, weightsInt, scale, zeroPoint}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		quant := &compute.Quantization{
			Scheme:    compute.QuantLinear,
			Scale:     params[2],
			ZeroPoint: params[3],
		}
		return f.FusedQuantizedDense(params[0], params[1], nil, quant, compute.ActivationNone)
	})
	if err != nil {
		t.Fatalf("FusedQuantizedDense failed: %+v", err)
	}
	// Dequantized weights = weightsInt * 0.1 = [[1, 2], [3, 4]]
	// x @ w = [[1, 2]] @ [[1, 2], [3, 4]] = [[7, 10]]
	want := [][]float32{{7, 10}}
	if ok, diff := testutil.IsInDelta(want, got, 1e-4); !ok {
		t.Errorf("Mismatch:\n%s", diff)
	}
}

func TestFusedScaledDotProductAttentionGQA(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %+v", err)
	}
	defer b.Finalize()

	// GQA: 2 query heads, 1 KV head (repeats=2)
	q := [][][][]float32{{{{1}, {1}}, {{1}, {1}}}} // [1, 2, 2, 1]
	k := [][][][]float32{{{{1}, {1}}}}             // [1, 1, 2, 1]
	v := [][][][]float32{{{{10}, {20}}}}           // [1, 1, 2, 1]

	got, err := testutil.Exec1(b, []any{q, k, v}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], compute.AttentionAxesLayoutBHSD, &compute.ScaledDotProductAttentionConfig{
			Scale:  1.0,
			Causal: true,
		})
		return out, err
	})
	if err != nil {
		t.Fatalf("GQA SDPA failed: %+v", err)
	}
	// Both query heads get identical output
	want := [][][][]float32{{{{10}, {15}}, {{10}, {15}}}}
	if ok, diff := testutil.IsInDelta(want, got, 1e-4); !ok {
		t.Errorf("GQA mismatch:\n%s", diff)
	}
}

func TestFusedLayerNormErrors(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %+v", err)
	}
	defer b.Finalize()

	builder := b.Builder("test_layernorm_err")
	mainFn := builder.Main()
	param, _ := mainFn.Parameter("x", shapes.Make(dtypes.Float32, 2, 3, 4), nil)

	// Non-contiguous axes [0, 2] should return NotImplemented error
	_, err = mainFn.FusedLayerNorm(param, []int{0, 2}, 1e-5, nil, nil)
	if err == nil || !compute.IsNotImplemented(err) {
		t.Errorf("Expected NotImplemented error for non-contiguous trailing axes, got: %v", err)
	}
}
