// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxruntime

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
	ort "github.com/yalue/onnxruntime_go"
)

type Executable struct {
	backend      *Backend
	session      *ort.DynamicAdvancedSession
	inputNames   []string
	inputShapes  []shapes.Shape
	outputNames  []string
	outputShapes []shapes.Shape
}

var _ compute.Executable = (*Executable)(nil)

func (e *Executable) Finalize() {
	if e.session != nil {
		_ = e.session.Destroy()
		e.session = nil
	}
}

func (e *Executable) Inputs() (names []string, inputShapes []shapes.Shape) {
	return e.inputNames, e.inputShapes
}

func (e *Executable) Outputs() (outputShapes []shapes.Shape) {
	return e.outputShapes
}

func (e *Executable) Execute(inputs []compute.Buffer, donate []bool, defaultDevice compute.DeviceNum) ([]compute.Buffer, error) {
	if e.session == nil {
		return nil, errors.New("cannot execute finalized executable")
	}

	isDummyInput := len(e.inputNames) == 1 && e.inputNames[0] == "dummy_input"
	expectedInputs := len(e.inputNames)
	if isDummyInput {
		expectedInputs = 0
	}

	if len(inputs) != expectedInputs {
		return nil, errors.Errorf("expected %d inputs, got %d", expectedInputs, len(inputs))
	}

	var ortInputs []ort.Value
	var dummyInputWrapper ortTensorWrapper

	if isDummyInput {
		wrapper, err := newOrtTensorWrapper(e.inputShapes[0], []int32{0})
		if err != nil {
			return nil, err
		}
		dummyInputWrapper = wrapper
		ortInputs = []ort.Value{wrapper.Value()}
	} else {
		ortInputs = make([]ort.Value, len(inputs))
		for i, inp := range inputs {
			buf, ok := inp.(*Buffer)
			if !ok {
				return nil, errors.Errorf("input %d is not a valid onnxruntime buffer", i)
			}
			if buf.wrapper == nil {
				return nil, errors.Errorf("input %d is finalized", i)
			}
			ortInputs[i] = buf.wrapper.Value()
		}
	}

	if dummyInputWrapper != nil {
		defer func() {
			_ = dummyInputWrapper.Destroy()
		}()
	}

	ortOutputs := make([]ort.Value, len(e.outputShapes))

	err := e.session.Run(ortInputs, ortOutputs)
	if err != nil {
		return nil, errors.Wrap(err, "onnxruntime execution failed")
	}

	outBuffers := make([]compute.Buffer, len(e.outputShapes))
	for i, sh := range e.outputShapes {
		wrapper, err := wrapOrtValue(ortOutputs[i], sh)
		if err != nil {
			for _, o := range ortOutputs {
				if o != nil {
					_ = o.Destroy()
				}
			}
			return nil, errors.Wrapf(err, "failed to wrap output %d", i)
		}
		outBuffers[i] = &Buffer{
			backend: e.backend,
			wrapper: wrapper,
			shape:   sh,
			device:  defaultDevice,
		}
	}

	return outBuffers, nil
}
