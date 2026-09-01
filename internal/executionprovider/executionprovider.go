// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package executionprovider defines the native GPU execution providers
// supported by the ONNX Runtime backend.
package executionprovider

import (
	"fmt"
	"strings"
)

// ExecutionProviderType identifies the ONNX Runtime execution provider used by a native backend.
type ExecutionProviderType int

const (
	// ExecutionProviderCPU runs on CPU only (the zero value).
	ExecutionProviderCPU ExecutionProviderType = iota
	// ExecutionProviderCUDA runs on an NVIDIA GPU via the CUDA execution provider.
	ExecutionProviderCUDA
	// ExecutionProviderMIGraphX runs on an AMD GPU via the MIGraphX execution provider.
	ExecutionProviderMIGraphX
)

// String returns the canonical name of the execution provider ("cpu", "cuda", or "migraphx").
func (t ExecutionProviderType) String() string {
	switch t {
	case ExecutionProviderCUDA:
		return "cuda"
	case ExecutionProviderMIGraphX:
		return "migraphx"
	default:
		return "cpu"
	}
}

// Parse maps a configuration token to an ExecutionProviderType.
// It accepts the same aliases as the backend config string: "cpu"/"" for CPU,
// "cuda"/"gpu" for CUDA, and "migraphx"/"rocm"/"amd" for MIGraphX.
func Parse(s string) (ExecutionProviderType, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "cpu":
		return ExecutionProviderCPU, nil
	case "cuda", "gpu":
		return ExecutionProviderCUDA, nil
	case "migraphx", "rocm", "amd":
		return ExecutionProviderMIGraphX, nil
	default:
		return ExecutionProviderCPU, fmt.Errorf("invalid execution provider %q: expected \"cpu\", \"cuda\", or \"migraphx\"", s)
	}
}
