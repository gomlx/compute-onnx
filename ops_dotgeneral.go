// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxruntime

import (
	"fmt"
	"strings"

	"github.com/gomlx/compute"
	onnx "github.com/gomlx/compute-onnx/internal/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapeinference"
	"github.com/pkg/errors"
)

// DotGeneral performs a general dot product (matrix multiplication) of lhs and rhs.
// It maps the generalized contracting and batching axes of XLA semantics to ONNX Einsum.
func (f *Function) DotGeneral(
	lhs compute.Value,
	lhsContractingAxes []int,
	lhsBatchAxes []int,
	rhs compute.Value,
	rhsContractingAxes []int,
	rhsBatchAxes []int,
	config compute.DotGeneralConfig,
) (compute.Value, error) {
	lhsNode, ok1 := lhs.(*Node)
	rhsNode, ok2 := rhs.(*Node)
	if !ok1 || !ok2 {
		return nil, errors.New("DotGeneral: inputs must be valid onnxruntime nodes")
	}

	// 1. Infer output shape
	outShape, err := shapeinference.DotGeneral(
		lhsNode.shape, lhsContractingAxes, lhsBatchAxes,
		rhsNode.shape, rhsContractingAxes, rhsBatchAxes,
		config,
	)
	if err != nil {
		return nil, err
	}

	// Determine accumulation and output types
	accumulationDType := lhsNode.shape.DType
	if config.AccumulatorDType != dtypes.InvalidDType {
		accumulationDType = config.AccumulatorDType
	}

	expectedOutputDType := lhsNode.shape.DType
	if config.OutputDType != dtypes.InvalidDType {
		expectedOutputDType = config.OutputDType
	}

	// Cast inputs to accumulator type if necessary
	lhsInput := lhsNode
	rhsInput := rhsNode

	if lhsNode.shape.DType != accumulationDType {
		casted, err := f.ConvertDType(lhsNode, accumulationDType)
		if err != nil {
			return nil, err
		}
		lhsInput = casted.(*Node)
	}

	if rhsNode.shape.DType != accumulationDType {
		casted, err := f.ConvertDType(rhsNode, accumulationDType)
		if err != nil {
			return nil, err
		}
		rhsInput = casted.(*Node)
	}

	// 2. Generate Einsum equation
	charPool := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
	charIdx := 0
	getNewChar := func() string {
		if charIdx >= len(charPool) {
			panic("DotGeneral: ran out of character pool for Einsum (rank is too high)")
		}
		c := string(charPool[charIdx])
		charIdx++
		return c
	}

	lhsRank := lhsInput.shape.Rank()
	rhsRank := rhsInput.shape.Rank()

	lhsChars := make([]string, lhsRank)
	for i := 0; i < lhsRank; i++ {
		lhsChars[i] = getNewChar()
	}

	rhsBatchMap := make(map[int]int)
	for idx, rhsAxis := range rhsBatchAxes {
		rhsBatchMap[rhsAxis] = lhsBatchAxes[idx]
	}

	rhsContractMap := make(map[int]int)
	for idx, rhsAxis := range rhsContractingAxes {
		rhsContractMap[rhsAxis] = lhsContractingAxes[idx]
	}

	rhsChars := make([]string, rhsRank)
	for i := 0; i < rhsRank; i++ {
		if lhsAxis, ok := rhsBatchMap[i]; ok {
			rhsChars[i] = lhsChars[lhsAxis]
		} else if lhsAxis, ok := rhsContractMap[i]; ok {
			rhsChars[i] = lhsChars[lhsAxis]
		} else {
			rhsChars[i] = getNewChar()
		}
	}

	outputChars := make([]string, 0)
	for _, lhsAxis := range lhsBatchAxes {
		outputChars = append(outputChars, lhsChars[lhsAxis])
	}

	lhsContractSet := make(map[int]bool)
	for _, axis := range lhsContractingAxes {
		lhsContractSet[axis] = true
	}
	lhsBatchSet := make(map[int]bool)
	for _, axis := range lhsBatchAxes {
		lhsBatchSet[axis] = true
	}
	for i := 0; i < lhsRank; i++ {
		if !lhsContractSet[i] && !lhsBatchSet[i] {
			outputChars = append(outputChars, lhsChars[i])
		}
	}

	rhsContractSet := make(map[int]bool)
	for _, axis := range rhsContractingAxes {
		rhsContractSet[axis] = true
	}
	rhsBatchSet := make(map[int]bool)
	for _, axis := range rhsBatchAxes {
		rhsBatchSet[axis] = true
	}
	for i := 0; i < rhsRank; i++ {
		if !rhsContractSet[i] && !rhsBatchSet[i] {
			outputChars = append(outputChars, rhsChars[i])
		}
	}

	equation := fmt.Sprintf("%s,%s->%s",
		strings.Join(lhsChars, ""),
		strings.Join(rhsChars, ""),
		strings.Join(outputChars, ""),
	)

	// 3. Create Einsum node
	einsumShape := outShape
	einsumShape.DType = accumulationDType

	f.nodeCount++
	einsumNode := &Node{
		name:   fmt.Sprintf("node_%d", f.nodeCount),
		opType: "Einsum",
		inputs: []*Node{lhsInput, rhsInput},
		shape:  einsumShape,
		attributes: []*onnx.AttributeProto{
			{
				Name: "equation",
				Type: onnx.AttributeProto_STRING,
				S:    []byte(equation),
			},
		},
	}
	f.nodes = append(f.nodes, einsumNode)

	var lastNode *Node = einsumNode
	if outShape.Rank() == 0 {
		newDimsConst, err := f.Constant([]int64{1}, 1)
		if err != nil {
			return nil, err
		}

		f.nodeCount++
		reshapeNode := &Node{
			name:   fmt.Sprintf("node_%d", f.nodeCount),
			opType: "Reshape",
			inputs: []*Node{einsumNode, newDimsConst.(*Node)},
			shape:  outShape,
		}
		f.nodes = append(f.nodes, reshapeNode)
		lastNode = reshapeNode
	}

	// Cast to final output type if necessary
	var finalVal compute.Value = lastNode
	if accumulationDType != expectedOutputDType {
		casted, err := f.ConvertDType(lastNode, expectedOutputDType)
		if err != nil {
			return nil, err
		}
		finalVal = casted
	}

	return finalVal, nil
}
