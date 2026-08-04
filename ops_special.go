// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"math"

	"github.com/gomlx/compute"
	onnx "github.com/gomlx/compute-onnx/internal/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func (f *Function) ConvertDType(operand compute.Value, targetDType dtypes.DType) (compute.Value, error) {
	xNode, ok := operand.(*Node)
	if !ok {
		return nil, errors.New("input must be a valid onnxruntime node")
	}

	outShape := xNode.shape
	outShape.DType = targetDType

	node := &Node{
		opType: "Cast",
		inputs: []*Node{xNode},
		shape:  outShape,
		attributes: []*onnx.AttributeProto{
			{
				Name: "to",
				Type: onnx.AttributeProto_INT,
				I:    int64(dtypeToONNX(targetDType)),
			},
		},
	}
	return f.addNode(node), nil
}

func (f *Function) Where(condition compute.Value, onTrue compute.Value, onFalse compute.Value) (compute.Value, error) {
	condNode, ok1 := condition.(*Node)
	onTrueNode, ok2 := onTrue.(*Node)
	onFalseNode, ok3 := onFalse.(*Node)
	if !ok1 || !ok2 || !ok3 {
		return nil, errors.New("inputs must be valid onnxruntime nodes")
	}

	outShape, err := shapeinference.Where(condNode.shape, onTrueNode.shape, onFalseNode.shape)
	if err != nil {
		return nil, err
	}

	node := &Node{
		opType: "Where",
		inputs: []*Node{condNode, onTrueNode, onFalseNode},
		shape:  outShape,
	}
	return f.addNode(node), nil
}

func (f *Function) Reshape(x compute.Value, newDimensions ...int) (compute.Value, error) {
	xNode, ok := x.(*Node)
	if !ok {
		return nil, errors.New("input must be a valid onnxruntime node")
	}

	outShape, err := shapeinference.Reshape(xNode.shape, newDimensions)
	if err != nil {
		return nil, err
	}

	newDims64 := make([]int64, len(newDimensions))
	for i, d := range newDimensions {
		newDims64[i] = int64(d)
	}

	shapeConstNode, err := f.Constant(newDims64, len(newDimensions))
	if err != nil {
		return nil, err
	}

	node := &Node{
		opType: "Reshape",
		inputs: []*Node{xNode, shapeConstNode.(*Node)},
		shape:  outShape,
		attributes: []*onnx.AttributeProto{
			{
				Name: "allowzero",
				Type: onnx.AttributeProto_INT,
				I:    1,
			},
		},
	}
	return f.addNode(node), nil
}

func (f *Function) Transpose(x compute.Value, permutation ...int) (compute.Value, error) {
	xNode, ok := x.(*Node)
	if !ok {
		return nil, errors.New("input must be a valid onnxruntime node")
	}

	outShape, err := shapeinference.Transpose(xNode.shape, permutation)
	if err != nil {
		return nil, err
	}

	perm64 := make([]int64, len(permutation))
	for i, p := range permutation {
		perm64[i] = int64(p)
	}

	node := &Node{
		opType: "Transpose",
		inputs: []*Node{xNode},
		shape:  outShape,
		attributes: []*onnx.AttributeProto{
			{
				Name: "perm",
				Type: onnx.AttributeProto_INTS,
				Ints: perm64,
			},
		},
	}
	return f.addNode(node), nil
}

func (f *Function) BroadcastInDim(x compute.Value, outputShape shapes.Shape, broadcastAxes []int) (compute.Value, error) {
	xNode, ok := x.(*Node)
	if !ok {
		return nil, errors.New("input must be a valid onnxruntime node")
	}

	reshapeDims := make([]int, outputShape.Rank())
	for i := range reshapeDims {
		reshapeDims[i] = 1
	}
	for i, axis := range broadcastAxes {
		reshapeDims[axis] = xNode.shape.Dimensions[i]
	}

	reshaped, err := f.Reshape(xNode, reshapeDims...)
	if err != nil {
		return nil, err
	}

	targetDims64 := make([]int64, outputShape.Rank())
	for i, d := range outputShape.Dimensions {
		targetDims64[i] = int64(d)
	}
	targetDimsConst, err := f.Constant(targetDims64, outputShape.Rank())
	if err != nil {
		return nil, err
	}

	node := &Node{
		opType: "Expand",
		inputs: []*Node{reshaped.(*Node), targetDimsConst.(*Node)},
		shape:  outputShape,
	}
	return f.addNode(node), nil
}

func (f *Function) Concatenate(axis int, inputs ...compute.Value) (compute.Value, error) {
	nodeInputs := make([]*Node, len(inputs))
	shapeInputs := make([]shapes.Shape, len(inputs))
	for i, inp := range inputs {
		n, ok := inp.(*Node)
		if !ok {
			return nil, errors.New("inputs must be valid onnxruntime nodes")
		}
		nodeInputs[i] = n
		shapeInputs[i] = n.shape
	}

	outShape, err := shapeinference.Concatenate(shapeInputs, axis)
	if err != nil {
		return nil, err
	}

	node := &Node{
		opType: "Concat",
		inputs: nodeInputs,
		shape:  outShape,
		attributes: []*onnx.AttributeProto{
			{
				Name: "axis",
				Type: onnx.AttributeProto_INT,
				I:    int64(axis),
			},
		},
	}
	return f.addNode(node), nil
}

func (f *Function) Slice(x compute.Value, start []int, limit []int, stride []int) (compute.Value, error) {
	xNode, ok := x.(*Node)
	if !ok {
		return nil, errors.New("input must be a valid onnxruntime node")
	}

	outShape, err := shapeinference.Slice(xNode.shape, start, limit, stride)
	if err != nil {
		return nil, err
	}

	rank := xNode.shape.Rank()
	starts64 := make([]int64, rank)
	ends64 := make([]int64, rank)
	axes64 := make([]int64, rank)
	steps64 := make([]int64, rank)

	for i := 0; i < rank; i++ {
		starts64[i] = int64(start[i])
		ends64[i] = int64(limit[i])
		axes64[i] = int64(i)
		steps64[i] = int64(stride[i])
	}

	startsConst, err := f.Constant(starts64, rank)
	if err != nil {
		return nil, err
	}
	endsConst, err := f.Constant(ends64, rank)
	if err != nil {
		return nil, err
	}
	axesConst, err := f.Constant(axes64, rank)
	if err != nil {
		return nil, err
	}
	stepsConst, err := f.Constant(steps64, rank)
	if err != nil {
		return nil, err
	}

	node := &Node{
		opType: "Slice",
		inputs: []*Node{xNode, startsConst.(*Node), endsConst.(*Node), axesConst.(*Node), stepsConst.(*Node)},
		shape:  outShape,
	}
	return f.addNode(node), nil
}

func (f *Function) Reverse(x compute.Value, axes ...int) (compute.Value, error) {
	xNode, ok := x.(*Node)
	if !ok {
		return nil, errors.New("input must be a valid onnxruntime node")
	}

	if len(axes) == 0 {
		return xNode, nil
	}

	starts := make([]int64, len(axes))
	ends := make([]int64, len(axes))
	steps := make([]int64, len(axes))
	axes64 := make([]int64, len(axes))

	for i, a := range axes {
		starts[i] = -1
		ends[i] = math.MinInt64
		steps[i] = -1
		axes64[i] = int64(a)
	}

	startsConst, err := f.Constant(starts, len(axes))
	if err != nil {
		return nil, err
	}
	endsConst, err := f.Constant(ends, len(axes))
	if err != nil {
		return nil, err
	}
	axesConst, err := f.Constant(axes64, len(axes))
	if err != nil {
		return nil, err
	}
	stepsConst, err := f.Constant(steps, len(axes))
	if err != nil {
		return nil, err
	}

	node := &Node{
		opType: "Slice",
		inputs: []*Node{xNode, startsConst.(*Node), endsConst.(*Node), axesConst.(*Node), stepsConst.(*Node)},
		shape:  xNode.shape,
	}
	return f.addNode(node), nil
}

func (f *Function) DynamicShape(operand compute.Value) (compute.Value, error) {
	xNode, ok := operand.(*Node)
	if !ok {
		return nil, errors.New("input must be a valid onnxruntime node")
	}

	rank := xNode.shape.Rank()

	if !xNode.shape.IsDynamic() {
		dims32 := make([]int32, rank)
		for i, d := range xNode.shape.Dimensions {
			dims32[i] = int32(d)
		}
		return f.Constant(dims32, rank)
	}

	shape64 := shapes.Make(dtypes.Int64, rank)
	shapeNode64 := f.addNode(&Node{
		opType: "Shape",
		inputs: []*Node{xNode},
		shape:  shape64,
	})

	return f.ConvertDType(shapeNode64, dtypes.Int32)
}

func (f *Function) DynamicDimensionSize(operand compute.Value, axis int) (compute.Value, error) {
	xNode, ok := operand.(*Node)
	if !ok {
		return nil, errors.New("input must be a valid onnxruntime node")
	}

	rank := xNode.shape.Rank()
	if axis < 0 || axis >= rank {
		return nil, errors.Errorf("axis %d out of bounds for rank %d", axis, rank)
	}

	if !xNode.shape.IsDynamic() || xNode.shape.Dimensions[axis] != shapes.DynamicDim {
		return f.Constant([]int32{int32(xNode.shape.Dimensions[axis])})
	}

	dynShape, err := f.DynamicShape(xNode)
	if err != nil {
		return nil, err
	}

	sliced, err := f.Slice(dynShape, []int{axis}, []int{axis + 1}, []int{1})
	if err != nil {
		return nil, err
	}

	return f.Reshape(sliced)
}

func (f *Function) DynamicReshape(operand compute.Value, dimensions ...compute.DynamicDimensionSpec) (compute.Value, error) {
	xNode, ok := operand.(*Node)
	if !ok {
		return nil, errors.New("input must be a valid onnxruntime node")
	}

	outShape, err := shapeinference.DynamicReshape(xNode.shape, dimensions)
	if err != nil {
		return nil, err
	}

	hasDynamicValue := false
	for _, spec := range dimensions {
		if spec.Value != nil {
			hasDynamicValue = true
			break
		}
	}

	if !outShape.IsDynamic() && !hasDynamicValue {
		return f.Reshape(xNode, outShape.Dimensions...)
	}

	parts := make([]compute.Value, len(dimensions))
	for i, spec := range dimensions {
		if spec.Name != "" && spec.Value == nil {
			constNode, err := f.Constant([]int64{-1}, 1)
			if err != nil {
				return nil, err
			}
			parts[i] = constNode
		} else if spec.Value != nil {
			valNode, ok := spec.Value.(*Node)
			if !ok {
				return nil, errors.New("dimension spec Value is not a valid onnxruntime node")
			}
			v := compute.Value(valNode)
			if valNode.shape.Rank() == 0 {
				var err error
				v, err = f.Reshape(v, 1)
				if err != nil {
					return nil, err
				}
			}
			currNode := v.(*Node)
			if currNode.shape.DType != dtypes.Int64 {
				var err error
				v, err = f.ConvertDType(v, dtypes.Int64)
				if err != nil {
					return nil, err
				}
			}
			parts[i] = v
		} else {
			constNode, err := f.Constant([]int64{int64(spec.Static)}, 1)
			if err != nil {
				return nil, err
			}
			parts[i] = constNode
		}
	}

	var shapeTensorNode compute.Value
	if len(parts) == 1 {
		shapeTensorNode = parts[0]
	} else {
		var err error
		shapeTensorNode, err = f.Concatenate(0, parts...)
		if err != nil {
			return nil, err
		}
	}

	shapeNode, ok := shapeTensorNode.(*Node)
	if !ok {
		return nil, errors.New("shape tensor is not a valid onnxruntime node")
	}

	node := &Node{
		opType: "Reshape",
		inputs: []*Node{xNode, shapeNode},
		shape:  outShape,
		attributes: []*onnx.AttributeProto{
			{
				Name: "allowzero",
				Type: onnx.AttributeProto_INT,
				I:    1,
			},
		},
	}
	return f.addNode(node), nil
}
