// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"fmt"
	"io"

	"github.com/gomlx/compute"
	"github.com/pkg/errors"
	"google.golang.org/protobuf/proto"
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

	onExec, ok := executable.(*Executable)
	if !ok {
		return errors.New("SaveModel: executable is not an ONNX executable (*onnxbackend.Executable)")
	}

	if onExec.backend != onBackend {
		return errors.New("SaveModel: executable was not created by the provided backend")
	}

	modelProto := onExec.ModelProto()
	if modelProto == nil {
		return errors.New("SaveModel: model ONNX proto not retained; backend.SetKeepModelProto(true) must be called prior to compilation")
	}

	if modelProto.Graph == nil {
		return errors.New("SaveModel: invalid ONNX model proto (nil Graph)")
	}

	graph := modelProto.Graph
	nameMap := make(map[string]string)

	// Remap input names
	for i, newName := range inputNames {
		if newName == "" {
			continue
		}
		var oldName string
		if i < len(graph.Input) {
			oldName = graph.Input[i].Name
			graph.Input[i].Name = newName
		} else {
			oldName = fmt.Sprintf("arg_%d", i)
		}
		if oldName != "" {
			nameMap[oldName] = newName
		}
	}

	// Remap output names
	for i, newName := range outputNames {
		if newName == "" {
			continue
		}
		var oldName string
		if i < len(graph.Output) {
			oldName = graph.Output[i].Name
			graph.Output[i].Name = newName
		} else {
			oldName = fmt.Sprintf("output_%d", i)
		}
		if oldName != "" {
			nameMap[oldName] = newName
		}
	}

	// Update node input/output tensor references if any names were mapped
	if len(nameMap) > 0 {
		for _, node := range graph.Node {
			for idx, inName := range node.Input {
				if replacement, ok := nameMap[inName]; ok {
					node.Input[idx] = replacement
				}
			}
			for idx, outName := range node.Output {
				if replacement, ok := nameMap[outName]; ok {
					node.Output[idx] = replacement
				}
			}
		}
	}

	modelBytes, err := proto.Marshal(modelProto)
	if err != nil {
		return errors.Wrap(err, "SaveModel: failed to marshal ONNX ModelProto")
	}

	_, err = w.Write(modelBytes)
	if err != nil {
		return errors.Wrap(err, "SaveModel: failed to write model bytes to writer")
	}

	return nil
}
