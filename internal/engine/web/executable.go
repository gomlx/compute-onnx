// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build js && wasm

package web

import (
	"syscall/js"

	"github.com/gomlx/compute"
	onnx "github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// Executable implements [compute.Executable] for ONNX Runtime Web.
type Executable struct {
	backend      compute.Backend
	session      *Session
	inputNames   []string
	inputShapes  []shapes.Shape
	outputNames  []string
	outputShapes []shapes.Shape
	modelProto   *onnx.ModelProto
}

var _ compute.Executable = (*Executable)(nil)

// NewExecutable creates a new web Executable.
func NewExecutable(backend compute.Backend, session *Session,
	inputNames []string, inputShapes []shapes.Shape,
	outputNames []string, outputShapes []shapes.Shape,
	modelProto *onnx.ModelProto) *Executable {

	return &Executable{
		backend:      backend,
		session:      session,
		inputNames:   inputNames,
		inputShapes:  inputShapes,
		outputNames:  outputNames,
		outputShapes: outputShapes,
		modelProto:   modelProto,
	}
}

// Backend returns the parent compute.Backend.
func (e *Executable) Backend() compute.Backend {
	return e.backend
}

// ModelProto returns the ONNX ModelProto struct if retain was enabled during compilation, or nil otherwise.
func (e *Executable) ModelProto() *onnx.ModelProto {
	return e.modelProto
}

// Inputs returns graph input names and shapes.
func (e *Executable) Inputs() (names []string, inputShapes []shapes.Shape) {
	return e.inputNames, e.inputShapes
}

// Outputs returns graph output shapes.
func (e *Executable) Outputs() (outputShapes []shapes.Shape) {
	return e.outputShapes
}

// Finalize releases the underlying session.
func (e *Executable) Finalize() {
	if e.session != nil {
		_ = e.session.Destroy()
		e.session = nil
	}
}

// Execute runs the executable with the provided inputs.
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

	global := js.Global()
	feeds := global.Get("Object").New()

	if isDummyInput {
		typedArray, err := ConvertSliceToTypedArray([]int32{0}, dtypes.Int32)
		if err != nil {
			return nil, err
		}
		dummyTensor, err := CreateJSTensor(dtypes.Int32, []int{1}, typedArray)
		if err != nil {
			return nil, err
		}
		feeds.Set("dummy_input", dummyTensor)
	} else {
		for i, inp := range inputs {
			buf, ok := inp.(*Buffer)
			if !ok {
				return nil, errors.Errorf("input %d is not a valid web Buffer", i)
			}
			if buf.wrapper == nil || buf.wrapper.jsTensor.IsUndefined() {
				return nil, errors.Errorf("input %d is finalized", i)
			}
			feeds.Set(e.inputNames[i], buf.wrapper.jsTensor)
		}
	}

	results, err := e.session.Run(feeds)
	if err != nil {
		return nil, err
	}

	outBuffers := make([]compute.Buffer, len(e.outputNames))
	for i, name := range e.outputNames {
		outTensor := results.Get(name)
		if outTensor.IsUndefined() || outTensor.IsNull() {
			return nil, errors.Errorf("output tensor %q missing from session run results", name)
		}

		sh := e.outputShapes[i]
		actualShape := sh
		if sh.IsDynamic() {
			dimsVal := outTensor.Get("dims")
			numDims := dimsVal.Length()
			concreteDims := make([]int, numDims)
			for d := 0; d < numDims; d++ {
				concreteDims[d] = dimsVal.Index(d).Int()
			}
			actualShape.Dimensions = concreteDims
		}

		wrapper := NewWebTensorWrapper(outTensor, actualShape)
		outBuffers[i] = NewBuffer(e.backend, wrapper, actualShape, defaultDevice, false, e)
	}

	// Donate inputs if requested
	for i, inp := range inputs {
		if i < len(donate) && donate[i] {
			_ = inp.Finalize()
		}
	}

	return outBuffers, nil
}
