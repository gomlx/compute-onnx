// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"fmt"
	"io"

	"github.com/gomlx/compute"
	ort "github.com/gomlx/compute-onnx/internal/ort"
	onnx "github.com/gomlx/compute-onnx/internal/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
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

// LoadModel loads an ONNX model from an io.Reader and compiles it into a runnable compute.Executable.
// It parses input and output tensor shapes and data types directly from the ONNX ModelProto
// and creates an ONNX Runtime session ready for execution.
func LoadModel(backend compute.Backend, r io.Reader) (compute.Executable, error) {
	onBackend, ok := backend.(*Backend)
	if !ok {
		return nil, errors.New("LoadModel: backend is not an ONNX backend (*onnxbackend.Backend)")
	}

	modelBytes, err := io.ReadAll(r)
	if err != nil {
		return nil, errors.Wrap(err, "LoadModel: failed to read model bytes")
	}

	modelProto := &onnx.ModelProto{}
	err = proto.Unmarshal(modelBytes, modelProto)
	if err != nil {
		return nil, errors.Wrap(err, "LoadModel: failed to unmarshal ONNX ModelProto")
	}

	if modelProto.Graph == nil {
		return nil, errors.New("LoadModel: invalid ONNX model proto (nil Graph)")
	}

	graph := modelProto.Graph

	// Build map of initializers to filter out initializer names from inputs
	initializers := make(map[string]bool, len(graph.Initializer))
	for _, init := range graph.Initializer {
		initializers[init.Name] = true
	}

	var inputNames []string
	var inputShapes []shapes.Shape
	for _, valInfo := range graph.Input {
		if initializers[valInfo.Name] {
			continue // Skip initializers declared in graph inputs
		}
		name := valInfo.Name
		shape := onnxValueInfoToShape(valInfo)
		inputNames = append(inputNames, name)
		inputShapes = append(inputShapes, shape)
	}

	var outputNames []string
	var outputShapes []shapes.Shape
	for _, valInfo := range graph.Output {
		name := valInfo.Name
		shape := onnxValueInfoToShape(valInfo)
		outputNames = append(outputNames, name)
		outputShapes = append(outputShapes, shape)
	}

	var options *ort.SessionOptions
	if options, err = ort.NewSessionOptions(); err != nil {
		return nil, errors.Wrap(err, "LoadModel: failed to create ONNX Runtime SessionOptions")
	}
	defer options.Destroy()

	if onBackend.cuda {
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return nil, errors.Wrap(err, "LoadModel: failed to create ONNX Runtime CUDAProviderOptions")
		}
		defer cudaOpts.Destroy()

		_ = cudaOpts.Update(map[string]string{
			"do_copy_in_default_stream": "1",
		})

		err = options.AppendExecutionProviderCUDA(cudaOpts)
		if err != nil {
			return nil, errors.Wrap(err, "LoadModel: failed to append CUDA execution provider to SessionOptions")
		}
	}

	logSev := onBackend.logSeverity
	if logSev < 0 {
		logSev = 3 // ORT_LOGGING_LEVEL_ERROR by default
	}
	err = options.SetSessionLogSeverityLevel(logSev)
	if err != nil {
		return nil, errors.Wrap(err, "LoadModel: failed to set ONNX Runtime session log severity level")
	}

	session, err := ort.NewDynamicAdvancedSessionWithONNXData(modelBytes, inputNames, outputNames, options)
	if err != nil {
		return nil, errors.Wrap(err, "LoadModel: failed to create ONNX Runtime session")
	}

	var savedModelProto *onnx.ModelProto
	if onBackend.keepModelProto {
		savedModelProto = modelProto
	}

	return newExecutable(onBackend, session, inputNames, inputShapes, outputNames, outputShapes, savedModelProto), nil
}

func onnxValueInfoToShape(valInfo *onnx.ValueInfoProto) shapes.Shape {
	if valInfo == nil || valInfo.Type == nil {
		return shapes.Invalid()
	}
	tensorType := valInfo.Type.GetTensorType()
	if tensorType == nil {
		return shapes.Invalid()
	}

	dtype := onnxDTypeToGoMLX(onnx.TensorProto_DataType(tensorType.ElemType))

	var dims []int
	var axisNames []string
	hasAxisNames := false
	if tensorType.Shape != nil {
		dims = make([]int, len(tensorType.Shape.Dim))
		axisNames = make([]string, len(tensorType.Shape.Dim))
		for i, d := range tensorType.Shape.Dim {
			if d.GetDimValue() > 0 {
				dims[i] = int(d.GetDimValue())
			} else {
				dims[i] = shapes.DynamicDim
				if param := d.GetDimParam(); param != "" {
					axisNames[i] = param
					hasAxisNames = true
				}
			}
		}
	}

	if hasAxisNames {
		return shapes.MakeDynamic(dtype, dims, axisNames)
	}
	return shapes.Make(dtype, dims...)
}

func onnxDTypeToGoMLX(dt onnx.TensorProto_DataType) dtypes.DType {
	switch dt {
	case onnx.TensorProto_FLOAT:
		return dtypes.Float32
	case onnx.TensorProto_DOUBLE:
		return dtypes.Float64
	case onnx.TensorProto_INT32:
		return dtypes.Int32
	case onnx.TensorProto_INT64:
		return dtypes.Int64
	case onnx.TensorProto_BOOL:
		return dtypes.Bool
	case onnx.TensorProto_INT8:
		return dtypes.Int8
	case onnx.TensorProto_UINT8:
		return dtypes.Uint8
	case onnx.TensorProto_INT16:
		return dtypes.Int16
	case onnx.TensorProto_UINT16:
		return dtypes.Uint16
	case onnx.TensorProto_UINT32:
		return dtypes.Uint32
	case onnx.TensorProto_UINT64:
		return dtypes.Uint64
	default:
		return dtypes.InvalidDType
	}
}
