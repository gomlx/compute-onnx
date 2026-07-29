// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxruntime

import (
	"os"
	"runtime"

	"github.com/gomlx/compute"
	onnx "github.com/gomlx/compute-onnx/internal/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
	ort "github.com/gomlx/compute-onnx/internal/ort"
	"google.golang.org/protobuf/proto"
	"k8s.io/klog/v2"
)

func shapeToONNX(shape shapes.Shape) *onnx.TensorShapeProto {
	dims := shape.Dimensions
	if len(dims) == 0 {
		dims = []int{1}
	}
	onnxDims := make([]*onnx.TensorShapeProto_Dimension, len(dims))
	for i, d := range dims {
		onnxDims[i] = &onnx.TensorShapeProto_Dimension{
			Value: &onnx.TensorShapeProto_Dimension_DimValue{
				DimValue: int64(d),
			},
		}
	}
	return &onnx.TensorShapeProto{
		Dim: onnxDims,
	}
}

func dtypeToONNX(dt dtypes.DType) onnx.TensorProto_DataType {
	switch dt {
	case dtypes.Float32:
		return onnx.TensorProto_FLOAT
	case dtypes.Float64:
		return onnx.TensorProto_DOUBLE
	case dtypes.Float16:
		return onnx.TensorProto_FLOAT
	case dtypes.BFloat16:
		return onnx.TensorProto_FLOAT
	case dtypes.Int32:
		return onnx.TensorProto_INT32
	case dtypes.Int64:
		return onnx.TensorProto_INT64
	case dtypes.Bool:
		return onnx.TensorProto_BOOL
	case dtypes.Int8:
		return onnx.TensorProto_INT8
	case dtypes.Uint8:
		return onnx.TensorProto_UINT8
	case dtypes.Int16:
		return onnx.TensorProto_INT16
	case dtypes.Uint16:
		return onnx.TensorProto_UINT16
	case dtypes.Uint32:
		return onnx.TensorProto_UINT32
	case dtypes.Uint64:
		return onnx.TensorProto_UINT64
	default:
		return onnx.TensorProto_UNDEFINED
	}
}

func constantToTensorProto(name string, shape shapes.Shape, flat any) *onnx.TensorProto {
	var rawBytes []byte
	var dt onnx.TensorProto_DataType

	switch shape.DType {
	case dtypes.Float16:
		f16Slice := flat.([]float16.Float16)
		f32Slice := float16ToFloat32(f16Slice)
		rawBytes = dtypes.UnsafeByteSlice(f32Slice)
		dt = onnx.TensorProto_FLOAT
	case dtypes.BFloat16:
		bf16Slice := flat.([]bfloat16.BFloat16)
		f32Slice := bfloat16ToFloat32(bf16Slice)
		rawBytes = dtypes.UnsafeByteSlice(f32Slice)
		dt = onnx.TensorProto_FLOAT
	default:
		rawBytes = dtypes.UnsafeByteSliceFromAny(flat)
		dt = dtypeToONNX(shape.DType)
	}

	dims := shape.Dimensions
	if len(dims) == 0 {
		dims = []int{1}
	}
	onnxDims := make([]int64, len(dims))
	for i, d := range dims {
		onnxDims[i] = int64(d)
	}
	return &onnx.TensorProto{
		DataType: int32(dt),
		Dims:     onnxDims,
		Name:     name,
		RawData:  rawBytes,
	}
}

func (b *Builder) Compile() (compute.Executable, error) {
	mainFn := b.mainFn
	if klog.V(2).Enabled() {
		klog.Infof("COMPILE: builder=%q, mainFn=%q, num_params=%d", b.name, mainFn.name, len(mainFn.params))
		for i, p := range mainFn.params {
			klog.Infof("  param[%d]: name=%s shape=%s", i, p.name, p.shape)
		}
	}

	// Create inputs list for ONNX graph
	inputs := make([]*onnx.ValueInfoProto, 0, len(mainFn.params))
	inputNames := make([]string, len(mainFn.params))
	inputShapes := make([]shapes.Shape, len(mainFn.params))
	for i, p := range mainFn.params {
		onnxType := &onnx.TypeProto{
			Value: &onnx.TypeProto_TensorType{
				TensorType: &onnx.TypeProto_Tensor{
					ElemType: int32(dtypeToONNX(p.shape.DType)),
					Shape:    shapeToONNX(p.shape),
				},
			},
		}
		inputs = append(inputs, &onnx.ValueInfoProto{
			Name: p.name,
			Type: onnxType,
		})
		inputNames[i] = p.name
		inputShapes[i] = p.shape
	}

	if len(mainFn.params) == 0 {
		dummyShape := shapes.Make(dtypes.Int32, 1)
		onnxType := &onnx.TypeProto{
			Value: &onnx.TypeProto_TensorType{
				TensorType: &onnx.TypeProto_Tensor{
					ElemType: int32(onnx.TensorProto_INT32),
					Shape:    shapeToONNX(dummyShape),
				},
			},
		}
		inputs = append(inputs, &onnx.ValueInfoProto{
			Name: "dummy_input",
			Type: onnxType,
		})
		inputNames = []string{"dummy_input"}
		inputShapes = []shapes.Shape{dummyShape}
	}

	// Create outputs list for ONNX graph
	outputs := make([]*onnx.ValueInfoProto, 0, len(mainFn.returns))
	outputNames := make([]string, len(mainFn.returns))
	outputShapes := make([]shapes.Shape, len(mainFn.returns))
	for i, r := range mainFn.returns {
		onnxType := &onnx.TypeProto{
			Value: &onnx.TypeProto_TensorType{
				TensorType: &onnx.TypeProto_Tensor{
					ElemType: int32(dtypeToONNX(r.shape.DType)),
					Shape:    shapeToONNX(r.shape),
				},
			},
		}
		outputs = append(outputs, &onnx.ValueInfoProto{
			Name: r.name,
			Type: onnxType,
		})
		outputNames[i] = r.name
		outputShapes[i] = r.shape
	}

	// Create nodes and initializers
	var onnxNodes []*onnx.NodeProto
	var onnxInitializers []*onnx.TensorProto

	for _, node := range mainFn.nodes {
		if node.opType == "Parameter" {
			continue
		}
		if node.opType == "Constant" {
			initProto := constantToTensorProto(node.name, node.shape, node.flatValue)
			onnxInitializers = append(onnxInitializers, initProto)
			continue
		}

		nodeInputs := make([]string, len(node.inputs))
		for i, inp := range node.inputs {
			nodeInputs[i] = inp.name
		}

		nodeOutputs := node.outputNames
		if len(nodeOutputs) == 0 {
			nodeOutputs = []string{node.name}
		}

		nodeProto := &onnx.NodeProto{
			Input:     nodeInputs,
			Output:    nodeOutputs,
			OpType:    node.opType,
			Domain:    node.domain,
			Attribute: node.attributes,
		}
		onnxNodes = append(onnxNodes, nodeProto)
	}

	graph := &onnx.GraphProto{
		Name:        b.name,
		Node:        onnxNodes,
		Initializer: onnxInitializers,
		Input:       inputs,
		Output:      outputs,
	}

	model := &onnx.ModelProto{
		IrVersion: 9,
		Graph:     graph,
		OpsetImport: []*onnx.OperatorSetIdProto{
			{
				Domain:  "",
				Version: 21,
			},
			{
				Domain:  "com.microsoft",
				Version: 1,
			},
		},
	}

	modelBytes, err := proto.Marshal(model)
	if err != nil {
		return nil, errors.Wrap(err, "failed to marshal ONNX ModelProto")
	}

	// Strategy A: Immediately release Go AST structs & backing slices and trigger GC
	// before creating the ONNX Runtime session.
	model = nil
	graph = nil
	onnxNodes = nil
	onnxInitializers = nil
	for _, node := range mainFn.nodes {
		node.flatValue = nil
	}
	runtime.GC()

	var options *ort.SessionOptions
	if options, err = ort.NewSessionOptions(); err != nil {
		return nil, errors.Wrap(err, "failed to create ONNX Runtime SessionOptions")
	}
	defer options.Destroy()

	if b.backend.cuda {
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return nil, errors.Wrap(err, "failed to create ONNX Runtime CUDAProviderOptions")
		}
		defer cudaOpts.Destroy()

		_ = cudaOpts.Update(map[string]string{
			"do_copy_in_default_stream": "1",
		})

		err = options.AppendExecutionProviderCUDA(cudaOpts)
		if err != nil {
			return nil, errors.Wrap(err, "failed to append CUDA execution provider to SessionOptions")
		}
	}

	logSev := b.backend.logSeverity
	if logSev < 0 {
		logSev = 3 // ORT_LOGGING_LEVEL_ERROR by default
	}
	err = options.SetSessionLogSeverityLevel(logSev)
	if err != nil {
		return nil, errors.Wrap(err, "failed to set ONNX Runtime session log severity level")
	}

	session, err := ort.NewDynamicAdvancedSessionWithONNXData(modelBytes, inputNames, outputNames, options)
	if err != nil {
		_ = os.WriteFile("failed_model.onnx", modelBytes, 0644)
		return nil, errors.Wrap(err, "failed to create ONNX Runtime session")
	}

	return newExecutable(b.backend, session, inputNames, inputShapes, outputNames, outputShapes), nil
}
