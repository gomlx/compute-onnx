// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !js

package native

import (
	"testing"

	"github.com/gomlx/compute-onnx/internal/executionprovider"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

func TestBufferIsShared(t *testing.T) {
	tests := []struct {
		ep         executionprovider.Type
		wantShared bool
	}{
		{ep: executionprovider.CPU, wantShared: true},
		{ep: executionprovider.CUDA, wantShared: false},
		{ep: executionprovider.MIGraphX, wantShared: true},
	}

	for _, tc := range tests {
		buf := NewBuffer(nil, nil, shapes.Make(dtypes.Float32, 1), 0, tc.ep, nil)
		if got := buf.IsShared(); got != tc.wantShared {
			t.Errorf("Buffer.IsShared() for EP %v = %v, want %v", tc.ep, got, tc.wantShared)
		}
	}
}
