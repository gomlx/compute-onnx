// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"io"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute-onnx/internal/engine/native"
	"github.com/gomlx/compute-onnx/internal/graph"
	onnx "github.com/gomlx/compute-onnx/support/protos"
	"github.com/pkg/errors"
)

// SaveModel exports the computation graph associated with the given executable as an ONNX model file/stream.
// It verifies that both backend and executable belong to the ONNX backend package.
// If inputNames or outputNames are provided (non-empty), they rename the graph's inputs and outputs
// and update all corresponding internal node edge references before saving.
// If inputNames or outputNames are nil or empty, default or pre-existing graph names (e.g., "arg_0", "output_0") are retained.
//
// Note: The backend must have had KeepModelProto set to true prior to compiling the executable,
// otherwise SaveModel will return an error indicating that the graph proto was discarded.
func SaveModel(backend compute.Backend, executable compute.Executable, w io.Writer, inputNames []string, outputNames []string) error {
	onBackend, ok := backend.(*Backend)
	if !ok {
		return errors.New("SaveModel: backend is not an ONNX backend (*onnxbackend.Backend)")
	}

	onExec, ok := executable.(*native.Executable)
	if !ok {
		return errors.New("SaveModel: executable is not an ONNX executable (*native.Executable)")
	}

	if onExec.Backend() != onBackend && onExec.Backend() != backend {
		return errors.New("SaveModel: executable was not created by the provided backend")
	}

	modelProto := onExec.ModelProto()
	if modelProto == nil {
		return errors.New("SaveModel: model ONNX proto not retained; backend.SetKeepModelProto(true) must be called prior to compilation")
	}

	modelBytes, err := graph.RemapAndMarshalModel(modelProto, inputNames, outputNames)
	if err != nil {
		return errors.Wrap(err, "SaveModel")
	}

	_, err = w.Write(modelBytes)
	if err != nil {
		return errors.Wrap(err, "SaveModel: failed to write model bytes to writer")
	}

	return nil
}

// LoadModel loads an ONNX model from an io.Reader and compiles it into a runnable compute.Executable.
// It parses input and output tensor shapes and data types directly from the ONNX ModelProto
// and creates an ONNX Runtime session ready for execution.
func LoadModel(backend compute.Backend, r io.Reader) (compute.Executable, error) {
	onBackend, ok := backend.(*Backend)
	if !ok {
		return nil, errors.New("LoadModel: backend is not an ONNX backend (*onnxbackend.Backend)")
	}

	modelProto, modelBytes, inputNames, inputShapes, outputNames, outputShapes, err := graph.ParseModelProto(r)
	if err != nil {
		return nil, errors.Wrap(err, "LoadModel")
	}

	session, err := native.CreateSession(modelBytes, inputNames, outputNames, onBackend.cuda, onBackend.logSeverity)
	if err != nil {
		return nil, errors.Wrap(err, "LoadModel: failed to create ONNX Runtime session")
	}

	var savedModelProto *onnx.ModelProto
	if onBackend.keepModelProto {
		savedModelProto = modelProto
	}

	return native.NewExecutable(onBackend, session, inputNames, inputShapes, outputNames, outputShapes, savedModelProto, onBackend.cuda), nil
}
