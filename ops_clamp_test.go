// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/support/testutil"
)

func TestClamp(t *testing.T) {
	tests := []struct {
		name string
		min  any
		x    any
		max  any
		want any
	}{
		{"float32 scalar bounds", float32(-1), []float32{-5, -1, 0, 2, 10}, float32(3), []float32{-1, -1, 0, 2, 3}},
		{"elementwise bounds", []float32{0, 2, 4}, []float32{-1, 3, 9}, []float32{1, 3, 5}, []float32{0, 3, 5}},
		{"scalar lower bound", float64(0), [][]float64{{-2, 3}, {4, 8}}, [][]float64{{1, 2}, {3, 5}}, [][]float64{{0, 2}, {3, 5}}},
		{"scalar upper bound", []int32{-3, -2, -1}, []int32{-9, 0, 8}, int32(4), []int32{-3, 0, 4}},
		{"int8 bounds", int8(-2), []int8{-128, -1, 127}, int8(2), []int8{-2, -1, 2}},
		{"uint8 bounds", uint8(2), []uint8{0, 3, 255}, uint8(4), []uint8{2, 3, 4}},
		{"scalar input", float32(0), float32(9), float32(3), float32(3)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := testutil.Exec1(backend, []any{test.min, test.x, test.max},
				func(f compute.Function, values []compute.Value) (compute.Value, error) {
					return f.Clamp(values[0], values[1], values[2])
				})
			if err != nil {
				t.Fatalf("Clamp failed: %v", err)
			}
			if ok, diff := testutil.IsEqual(test.want, got); !ok {
				t.Errorf("Clamp mismatch:\n%s", diff)
			}
		})
	}
}
