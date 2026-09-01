// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package executionprovider defines the native GPU execution providers
// supported by the ONNX Runtime backend.
package executionprovider

import (
	"fmt"
	"strings"
)

// Type identifies the ONNX Runtime execution provider used by a native backend.
type Type int

const (
	// CPU runs on CPU only (the zero value).
	CPU Type = iota
	// CUDA runs on an NVIDIA GPU via the CUDA execution provider.
	CUDA
	// MIGraphX runs on an AMD GPU via the MIGraphX execution provider. It replaces the older ROCm EP.
	MIGraphX
)

// String returns the canonical name of the execution provider ("cpu", "cuda", or "migraphx").
func (t Type) String() string {
	switch t {
	case CUDA:
		return "cuda"
	case MIGraphX:
		return "migraphx"
	default:
		return "cpu"
	}
}

// Parse maps a configuration token to an executionprovider.Type.
// It accepts the same aliases as the backend config string: "cpu"/"" for CPU,
// "cuda"/"gpu" for CUDA, and "migraphx"/"rocm"/"amd" for MIGraphX.
func Parse(s string) (Type, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "cpu":
		return CPU, nil
	case "cuda", "gpu":
		return CUDA, nil
	case "migraphx", "rocm", "amd":
		return MIGraphX, nil
	default:
		return CPU, fmt.Errorf("invalid execution provider %q: expected \"cpu\", \"cuda\", or \"migraphx\"", s)
	}
}
