// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"bytes"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/stretchr/testify/require"
)

func TestSaveModel(t *testing.T) {
	backend, err := New("")
	require.NoError(t, err)
	defer backend.Finalize()

	onBackend, ok := backend.(*Backend)
	require.True(t, ok)

	// Step 1: Verify SaveModel returns error when SetKeepModelProto is false (default)
	builder := backend.Builder("test_save_disabled")
	fn := builder.Main()
	x, err := fn.Parameter("x", shapes.Make(dtypes.Float32, 2, 3), nil)
	require.NoError(t, err)
	sum, err := fn.Add(x, x)
	require.NoError(t, err)
	err = fn.Return([]compute.Value{sum}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)
	defer exec.Finalize()

	var buf bytes.Buffer
	err = SaveModel(backend, exec, &buf, nil, nil)
	require.Error(t, err)
	require.Contains(t, err.Error(), "backend.SetKeepModelProto(true) must be called")

	// Step 2: Enable SetKeepModelProto and save model with default input/output names
	onBackend.SetKeepModelProto(true)
	require.True(t, onBackend.KeepModelProto())

	builder2 := backend.Builder("test_save_enabled")
	fn2 := builder2.Main()
	p1, err := fn2.Parameter("in_a", shapes.Make(dtypes.Float32, 2, 2), nil)
	require.NoError(t, err)
	p2, err := fn2.Parameter("in_b", shapes.Make(dtypes.Float32, 2, 2), nil)
	require.NoError(t, err)
	sum2, err := fn2.Add(p1, p2)
	require.NoError(t, err)
	err = fn2.Return([]compute.Value{sum2}, nil)
	require.NoError(t, err)

	exec2, err := builder2.Compile()
	require.NoError(t, err)
	defer exec2.Finalize()

	buf.Reset()
	err = SaveModel(backend, exec2, &buf, nil, nil)
	require.NoError(t, err)
	require.NotEmpty(t, buf.Bytes())

	// Step 3: Save model with custom input and output names
	var bufCustom bytes.Buffer
	customInputs := []string{"custom_x", "custom_y"}
	customOutputs := []string{"custom_sum"}
	err = SaveModel(backend, exec2, &bufCustom, customInputs, customOutputs)
	require.NoError(t, err)
	require.NotEmpty(t, bufCustom.Bytes())

	// Verify the saved model bytes can be loaded into an ONNX Runtime session
	onExec2, ok := exec2.(*Executable)
	require.True(t, ok)
	require.NotNil(t, onExec2.ModelProto())
	require.Equal(t, "custom_x", onExec2.ModelProto().Graph.Input[0].Name)
	require.Equal(t, "custom_y", onExec2.ModelProto().Graph.Input[1].Name)
	require.Equal(t, "custom_sum", onExec2.ModelProto().Graph.Output[0].Name)
}
