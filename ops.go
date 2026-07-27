// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxruntime

import (
	"fmt"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapeinference"
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

	// Bitwise & Shift
	registerOp(compute.OpTypeBitwiseAnd)
	registerOp(compute.OpTypeBitwiseOr)
	registerOp(compute.OpTypeBitwiseXor)
	registerOp(compute.OpTypeShiftLeft)
	registerOp(compute.OpTypeShiftRightLogical)

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
	registerOp(compute.OpTypeSign)
	registerOp(compute.OpTypeLog1p)

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
	registerOp(compute.OpTypeArgMinMax)
	registerOp(compute.OpTypeReduceWindow)

	// DotGeneral / ConvGeneral
	registerOp(compute.OpTypeDotGeneral)
	registerOp(compute.OpTypeConvGeneral)

	// Scatter
	registerOp(compute.OpTypeScatterSum)
	registerOp(compute.OpTypeScatterMax)
	registerOp(compute.OpTypeScatterMin)

	// RNG
	registerOp(compute.OpTypeRNGBitGenerator)
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
