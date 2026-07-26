// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxruntime

import (
	"fmt"
	"math"

	"github.com/gomlx/compute"
	onnx "github.com/gomlx/compute-onnx/internal/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	// Binary
	registerOp(compute.OpTypeAdd)
	registerOp(compute.OpTypeSub)
	registerOp(compute.OpTypeMul)
	registerOp(compute.OpTypeDiv)
	registerOp(compute.OpTypeMax)
	registerOp(compute.OpTypeMin)
	registerOp(compute.OpTypePow)

	// Comparisons
	registerOp(compute.OpTypeEqual)
	registerOp(compute.OpTypeNotEqual)
	registerOp(compute.OpTypeGreaterThan)
	registerOp(compute.OpTypeGreaterOrEqual)
	registerOp(compute.OpTypeLessThan)
	registerOp(compute.OpTypeLessOrEqual)

	// Unary
	registerOp(compute.OpTypeAbs)
	registerOp(compute.OpTypeNeg)
	registerOp(compute.OpTypeCeil)
	registerOp(compute.OpTypeFloor)
	registerOp(compute.OpTypeRound)
	registerOp(compute.OpTypeSqrt)
	registerOp(compute.OpTypeExp)
	registerOp(compute.OpTypeLog)
	registerOp(compute.OpTypeCos)
	registerOp(compute.OpTypeSin)
	registerOp(compute.OpTypeTanh)
	registerOp(compute.OpTypeLogistic)
	registerOp(compute.OpTypeErf)

	// Logical
	registerOp(compute.OpTypeLogicalAnd)
	registerOp(compute.OpTypeLogicalOr)
	registerOp(compute.OpTypeLogicalXor)
	registerOp(compute.OpTypeLogicalNot)

	// Conversions
	registerOp(compute.OpTypeConvertDType)

	// Special Ops
	registerOp(compute.OpTypeIdentity)
	registerOp(compute.OpTypeWhere)
	registerOp(compute.OpTypeReshape)
	registerOp(compute.OpTypeReverse)
	registerOp(compute.OpTypeTranspose)
	registerOp(compute.OpTypeBroadcastInDim)
	registerOp(compute.OpTypeConcatenate)
	registerOp(compute.OpTypeSlice)
	registerOp(compute.OpTypeIota)

	// Reductions
	registerOp(compute.OpTypeReduceMin)
	registerOp(compute.OpTypeReduceMax)
	registerOp(compute.OpTypeReduceSum)
	registerOp(compute.OpTypeReduceProduct)
	registerOp(compute.OpTypeReduceLogicalAnd)
	registerOp(compute.OpTypeReduceLogicalOr)
	registerOp(compute.OpTypeReduceLogicalXor)
}

func (f *Function) addBinaryOp(opType compute.OpType, onnxOpType string, lhs, rhs compute.Value) (compute.Value, error) {
	lhsNode, ok1 := lhs.(*Node)
	rhsNode, ok2 := rhs.(*Node)
	if !ok1 || !ok2 {
		return nil, errors.New("inputs must be valid onnxruntime nodes")
	}

	outShape, err := shapeinference.BinaryOp(opType, lhsNode.shape, rhsNode.shape)
	if err != nil {
		return nil, err
	}

	f.nodeCount++
	node := &Node{
		name:   fmt.Sprintf("node_%d", f.nodeCount),
		opType: onnxOpType,
		inputs: []*Node{lhsNode, rhsNode},
		shape:  outShape,
	}
	f.nodes = append(f.nodes, node)
	return node, nil
}

func (f *Function) addComparisonOp(opType compute.OpType, onnxOpType string, lhs, rhs compute.Value) (compute.Value, error) {
	lhsNode, ok1 := lhs.(*Node)
	rhsNode, ok2 := rhs.(*Node)
	if !ok1 || !ok2 {
		return nil, errors.New("inputs must be valid onnxruntime nodes")
	}

	outShape, err := shapeinference.ComparisonOp(opType, lhsNode.shape, rhsNode.shape)
	if err != nil {
		return nil, err
	}

	f.nodeCount++
	node := &Node{
		name:   fmt.Sprintf("node_%d", f.nodeCount),
		opType: onnxOpType,
		inputs: []*Node{lhsNode, rhsNode},
		shape:  outShape,
	}
	f.nodes = append(f.nodes, node)
	return node, nil
}

func (f *Function) addUnaryOp(opType compute.OpType, onnxOpType string, x compute.Value) (compute.Value, error) {
	xNode, ok := x.(*Node)
	if !ok {
		return nil, errors.New("input must be a valid onnxruntime node")
	}

	outShape, err := shapeinference.UnaryOp(opType, xNode.shape)
	if err != nil {
		return nil, err
	}

	f.nodeCount++
	node := &Node{
		name:   fmt.Sprintf("node_%d", f.nodeCount),
		opType: onnxOpType,
		inputs: []*Node{xNode},
		shape:  outShape,
	}
	f.nodes = append(f.nodes, node)
	return node, nil
}

// Binary Ops

func (f *Function) Add(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeAdd, "Add", lhs, rhs)
}

func (f *Function) Sub(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeSub, "Sub", lhs, rhs)
}

func (f *Function) Mul(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeMul, "Mul", lhs, rhs)
}

func (f *Function) Div(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeDiv, "Div", lhs, rhs)
}

func (f *Function) Max(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeMax, "Max", lhs, rhs)
}

func (f *Function) Min(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeMin, "Min", lhs, rhs)
}

func (f *Function) Pow(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypePow, "Pow", lhs, rhs)
}

// Comparison Ops

func (f *Function) Equal(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addComparisonOp(compute.OpTypeEqual, "Equal", lhs, rhs)
}

func (f *Function) NotEqual(lhs, rhs compute.Value) (compute.Value, error) {
	eq, err := f.Equal(lhs, rhs)
	if err != nil {
		return nil, err
	}
	return f.LogicalNot(eq)
}

func (f *Function) GreaterThan(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addComparisonOp(compute.OpTypeGreaterThan, "Greater", lhs, rhs)
}

func (f *Function) GreaterOrEqual(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addComparisonOp(compute.OpTypeGreaterOrEqual, "GreaterOrEqual", lhs, rhs)
}

func (f *Function) LessThan(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addComparisonOp(compute.OpTypeLessThan, "Less", lhs, rhs)
}

func (f *Function) LessOrEqual(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addComparisonOp(compute.OpTypeLessOrEqual, "LessOrEqual", lhs, rhs)
}

// Unary Ops

func (f *Function) Abs(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeAbs, "Abs", x)
}

func (f *Function) Neg(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeNeg, "Neg", x)
}

func (f *Function) Ceil(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeCeil, "Ceil", x)
}

func (f *Function) Floor(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeFloor, "Floor", x)
}

func (f *Function) Round(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeRound, "Round", x)
}

func (f *Function) Sqrt(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeSqrt, "Sqrt", x)
}

func (f *Function) Exp(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeExp, "Exp", x)
}

func (f *Function) Log(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeLog, "Log", x)
}

func (f *Function) Cos(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeCos, "Cos", x)
}

func (f *Function) Sin(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeSin, "Sin", x)
}

func (f *Function) Tanh(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeTanh, "Tanh", x)
}

func (f *Function) Logistic(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeLogistic, "Sigmoid", x)
}

func (f *Function) Erf(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeErf, "Erf", x)
}

// Logical Ops

func (f *Function) LogicalAnd(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeLogicalAnd, "And", lhs, rhs)
}

func (f *Function) LogicalOr(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeLogicalOr, "Or", lhs, rhs)
}

func (f *Function) LogicalXor(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeLogicalXor, "Xor", lhs, rhs)
}

func (f *Function) LogicalNot(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeLogicalNot, "Not", x)
}

func (f *Function) ConvertDType(operand compute.Value, targetDType dtypes.DType) (compute.Value, error) {
	xNode, ok := operand.(*Node)
	if !ok {
		return nil, errors.New("input must be a valid onnxruntime node")
	}

	outShape := xNode.shape
	outShape.DType = targetDType

	f.nodeCount++
	node := &Node{
		name:   fmt.Sprintf("node_%d", f.nodeCount),
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
	f.nodes = append(f.nodes, node)
	return node, nil
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

	f.nodeCount++
	node := &Node{
		name:   fmt.Sprintf("node_%d", f.nodeCount),
		opType: "Where",
		inputs: []*Node{condNode, onTrueNode, onFalseNode},
		shape:  outShape,
	}
	f.nodes = append(f.nodes, node)
	return node, nil
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

	f.nodeCount++
	node := &Node{
		name:   fmt.Sprintf("node_%d", f.nodeCount),
		opType: "Reshape",
		inputs: []*Node{xNode, shapeConstNode.(*Node)},
		shape:  outShape,
	}
	f.nodes = append(f.nodes, node)
	return node, nil
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

	f.nodeCount++
	node := &Node{
		name:   fmt.Sprintf("node_%d", f.nodeCount),
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
	f.nodes = append(f.nodes, node)
	return node, nil
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

	f.nodeCount++
	node := &Node{
		name:   fmt.Sprintf("node_%d", f.nodeCount),
		opType: "Expand",
		inputs: []*Node{reshaped.(*Node), targetDimsConst.(*Node)},
		shape:  outputShape,
	}
	f.nodes = append(f.nodes, node)
	return node, nil
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

	f.nodeCount++
	node := &Node{
		name:   fmt.Sprintf("node_%d", f.nodeCount),
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
	f.nodes = append(f.nodes, node)
	return node, nil
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

	f.nodeCount++
	node := &Node{
		name:   fmt.Sprintf("node_%d", f.nodeCount),
		opType: "Slice",
		inputs: []*Node{xNode, startsConst.(*Node), endsConst.(*Node), axesConst.(*Node), stepsConst.(*Node)},
		shape:  outShape,
	}
	f.nodes = append(f.nodes, node)
	return node, nil
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

	f.nodeCount++
	node := &Node{
		name:   fmt.Sprintf("node_%d", f.nodeCount),
		opType: "Slice",
		inputs: []*Node{xNode, startsConst.(*Node), endsConst.(*Node), axesConst.(*Node), stepsConst.(*Node)},
		shape:  xNode.shape,
	}
	f.nodes = append(f.nodes, node)
	return node, nil
}

func (f *Function) Reduce(x compute.Value, opType compute.OpType, axes []int, keepDims bool) (compute.Value, error) {
	xNode, ok := x.(*Node)
	if !ok {
		return nil, errors.New("input must be a valid onnxruntime node")
	}

	originalDType := xNode.shape.DType
	isUnsigned := originalDType == dtypes.Uint8 || originalDType == dtypes.Uint16 || originalDType == dtypes.Uint32 || originalDType == dtypes.Uint64

	var reduceInput *Node
	if isUnsigned {
		castInput, err := f.ConvertDType(xNode, dtypes.Int64)
		if err != nil {
			return nil, err
		}
		reduceInput = castInput.(*Node)
	} else {
		reduceInput = xNode
	}

	outShape, err := shapeinference.Reduce(reduceInput.shape, axes)
	if err != nil {
		return nil, err
	}

	if keepDims {
		outShape.Dimensions = make([]int, reduceInput.shape.Rank())
		for axis, dim := range reduceInput.shape.Dimensions {
			reduced := false
			for _, a := range axes {
				if a == axis {
					reduced = true
					break
				}
			}
			if reduced {
				outShape.Dimensions[axis] = 1
			} else {
				outShape.Dimensions[axis] = dim
			}
		}
		if reduceInput.shape.AxisNames != nil {
			outShape.AxisNames = make([]string, reduceInput.shape.Rank())
			copy(outShape.AxisNames, reduceInput.shape.AxisNames)
		}
	}

	var ortOpType string
	switch opType {
	case compute.OpTypeReduceMin:
		ortOpType = "ReduceMin"
	case compute.OpTypeReduceMax:
		ortOpType = "ReduceMax"
	case compute.OpTypeReduceSum:
		ortOpType = "ReduceSum"
	case compute.OpTypeReduceProduct:
		ortOpType = "ReduceProd"
	default:
		return nil, errors.Errorf("unsupported reduction operation %s", opType)
	}

	axes64 := make([]int64, len(axes))
	for i, a := range axes {
		axes64[i] = int64(a)
	}
	axesConst, err := f.Constant(axes64, len(axes))
	if err != nil {
		return nil, err
	}

	keepDimsVal := int64(0)
	if keepDims {
		keepDimsVal = 1
	}

	f.nodeCount++
	reduceNode := &Node{
		name:   fmt.Sprintf("node_%d", f.nodeCount),
		opType: ortOpType,
		inputs: []*Node{reduceInput, axesConst.(*Node)},
		shape:  outShape,
		attributes: []*onnx.AttributeProto{
			{
				Name: "keepdims",
				Type: onnx.AttributeProto_INT,
				I:    keepDimsVal,
			},
		},
	}
	f.nodes = append(f.nodes, reduceNode)

	var finalNode *Node = reduceNode
	if !keepDims && outShape.Rank() == 0 {
		newDimsConst, err := f.Constant([]int64{1}, 1)
		if err != nil {
			return nil, err
		}

		f.nodeCount++
		reshapeNode := &Node{
			name:   fmt.Sprintf("node_%d", f.nodeCount),
			opType: "Reshape",
			inputs: []*Node{reduceNode, newDimsConst.(*Node)},
			shape:  outShape,
		}
		f.nodes = append(f.nodes, reshapeNode)
		finalNode = reshapeNode
	}

	if isUnsigned {
		castBack, err := f.ConvertDType(finalNode, originalDType)
		if err != nil {
			return nil, err
		}
		return castBack, nil
	}

	return finalNode, nil
}

func (f *Function) ReduceLogical(x compute.Value, opType compute.OpType, axes []int, keepDims bool) (compute.Value, error) {
	xNode, ok := x.(*Node)
	if !ok {
		return nil, errors.New("input must be a valid onnxruntime node")
	}

	xInt32, err := f.ConvertDType(xNode, dtypes.Int32)
	if err != nil {
		return nil, err
	}

	var reduced compute.Value
	switch opType {
	case compute.OpTypeReduceLogicalAnd:
		reduced, err = f.Reduce(xInt32, compute.OpTypeReduceProduct, axes, keepDims)
	case compute.OpTypeReduceLogicalOr:
		reduced, err = f.Reduce(xInt32, compute.OpTypeReduceMax, axes, keepDims)
	case compute.OpTypeReduceLogicalXor:
		sum, err := f.Reduce(xInt32, compute.OpTypeReduceSum, axes, keepDims)
		if err != nil {
			return nil, err
		}
		twoConst, err := f.Constant([]int32{2}, 1)
		if err != nil {
			return nil, err
		}
		div2, err := f.Div(sum, twoConst)
		if err != nil {
			return nil, err
		}
		mul2, err := f.Mul(div2, twoConst)
		if err != nil {
			return nil, err
		}
		mod2, err := f.Sub(sum, mul2)
		if err != nil {
			return nil, err
		}
		zeroConst, err := f.Constant([]int32{0}, 1)
		if err != nil {
			return nil, err
		}
		reduced, err = f.NotEqual(mod2, zeroConst)
	default:
		return nil, errors.Errorf("unsupported logical reduction type %s", opType)
	}
	if err != nil {
		return nil, err
	}

	if opType == compute.OpTypeReduceLogicalXor {
		return reduced, nil
	}
	return f.ConvertDType(reduced, dtypes.Bool)
}

func makeIotaFlatData(dtype dtypes.DType, n int) any {
	switch dtype {
	case dtypes.Float32:
		res := make([]float32, n)
		for i := 0; i < n; i++ {
			res[i] = float32(i)
		}
		return res
	case dtypes.Float64:
		res := make([]float64, n)
		for i := 0; i < n; i++ {
			res[i] = float64(i)
		}
		return res
	case dtypes.Int32:
		res := make([]int32, n)
		for i := 0; i < n; i++ {
			res[i] = int32(i)
		}
		return res
	case dtypes.Int64:
		res := make([]int64, n)
		for i := 0; i < n; i++ {
			res[i] = int64(i)
		}
		return res
	case dtypes.Int16:
		res := make([]int16, n)
		for i := 0; i < n; i++ {
			res[i] = int16(i)
		}
		return res
	case dtypes.Int8:
		res := make([]int8, n)
		for i := 0; i < n; i++ {
			res[i] = int8(i)
		}
		return res
	case dtypes.Uint8:
		res := make([]uint8, n)
		for i := 0; i < n; i++ {
			res[i] = uint8(i)
		}
		return res
	case dtypes.Uint16:
		res := make([]uint16, n)
		for i := 0; i < n; i++ {
			res[i] = uint16(i)
		}
		return res
	case dtypes.Uint32:
		res := make([]uint32, n)
		for i := 0; i < n; i++ {
			res[i] = uint32(i)
		}
		return res
	case dtypes.Uint64:
		res := make([]uint64, n)
		for i := 0; i < n; i++ {
			res[i] = uint64(i)
		}
		return res
	case dtypes.Float16:
		res := make([]float16.Float16, n)
		for i := 0; i < n; i++ {
			res[i] = float16.FromFloat32(float32(i))
		}
		return res
	case dtypes.BFloat16:
		res := make([]bfloat16.BFloat16, n)
		for i := 0; i < n; i++ {
			res[i] = bfloat16.FromFloat32(float32(i))
		}
		return res
	default:
		return nil
	}
}

func (f *Function) Iota(shape shapes.Shape, iotaAxis int) (compute.Value, error) {
	n := shape.Dimensions[iotaAxis]
	flatData := makeIotaFlatData(shape.DType, n)
	if flatData == nil {
		return nil, errors.Errorf("unsupported DType %s for Iota", shape.DType)
	}

	constNode, err := f.Constant(flatData, n)
	if err != nil {
		return nil, err
	}

	return f.BroadcastInDim(constNode, shape, []int{iotaAxis})
}

func (f *Function) ReduceMin(x compute.Value, axes ...int) (compute.Value, error) {
	return f.Reduce(x, compute.OpTypeReduceMin, axes, false)
}

func (f *Function) ReduceMax(x compute.Value, axes ...int) (compute.Value, error) {
	return f.Reduce(x, compute.OpTypeReduceMax, axes, false)
}

func (f *Function) ReduceSum(x compute.Value, axes ...int) (compute.Value, error) {
	return f.Reduce(x, compute.OpTypeReduceSum, axes, false)
}

func (f *Function) ReduceProduct(x compute.Value, axes ...int) (compute.Value, error) {
	return f.Reduce(x, compute.OpTypeReduceProduct, axes, false)
}

func (f *Function) ReduceLogicalAnd(x compute.Value, axes ...int) (compute.Value, error) {
	return f.ReduceLogical(x, compute.OpTypeReduceLogicalAnd, axes, false)
}

func (f *Function) ReduceLogicalOr(x compute.Value, axes ...int) (compute.Value, error) {
	return f.ReduceLogical(x, compute.OpTypeReduceLogicalOr, axes, false)
}

func (f *Function) ReduceLogicalXor(x compute.Value, axes ...int) (compute.Value, error) {
	return f.ReduceLogical(x, compute.OpTypeReduceLogicalXor, axes, false)
}
