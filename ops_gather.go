// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxruntime

import (
	"fmt"

	"github.com/gomlx/compute"
	onnx "github.com/gomlx/compute-onnx/internal/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	registerOp(compute.OpTypeGather)
}

// Gather gathers slices from operand using coordinates specified by startIndices.
// It maps the generalized XLA Gather semantics to ONNX GatherND, Reshape, and Transpose.
func (f *Function) Gather(
	operand compute.Value,
	startIndices compute.Value,
	indexVectorAxis int,
	offsetOutputAxes []int,
	collapsedSliceAxes []int,
	startIndexMap []int,
	sliceSizes []int,
	indicesAreSorted bool,
) (compute.Value, error) {
	operandNode, ok1 := operand.(*Node)
	startIndicesNode, ok2 := startIndices.(*Node)
	if !ok1 || !ok2 {
		return nil, errors.New("Gather: inputs must be valid onnxruntime nodes")
	}

	operandShape := operandNode.shape
	startIndicesShape := startIndicesNode.shape
	Q := startIndicesShape.Rank()

	// 1. Prepare indices to move the indexVectorAxis to the last axis.
	var indicesNode *Node
	if indexVectorAxis == Q {
		// Virtual indexVectorAxis of size 1. Append a dimension of size 1.
		newDims := make([]int, Q+1)
		copy(newDims, startIndicesShape.Dimensions)
		newDims[Q] = 1
		indicesNodeVal, err := f.Reshape(startIndicesNode, newDims...)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: failed to reshape startIndices to append virtual vector axis")
		}
		indicesNode = indicesNodeVal.(*Node)
	} else if indexVectorAxis == Q-1 {
		// Already at the end.
		indicesNode = startIndicesNode
	} else {
		// Move indexVectorAxis to the end.
		perm := make([]int, Q)
		idx := 0
		for i := 0; i < Q; i++ {
			if i != indexVectorAxis {
				perm[idx] = i
				idx++
			}
		}
		perm[Q-1] = indexVectorAxis
		indicesNodeVal, err := f.Transpose(startIndicesNode, perm...)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: failed to transpose startIndices to move vector axis to end")
		}
		indicesNode = indicesNodeVal.(*Node)
	}

	if indicesNode.shape.DType != dtypes.Int64 {
		castedIndices, err := f.ConvertDType(indicesNode, dtypes.Int64)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: failed to cast indices to Int64")
		}
		indicesNode = castedIndices.(*Node)
	}

	V := indicesNode.shape.Dimensions[indicesNode.shape.Rank()-1]

	// 2. Prepare operand by transposing the mapped dimensions to the front.
	mappedSet := make(map[int]bool)
	for _, axis := range startIndexMap {
		mappedSet[axis] = true
	}
	var unmapped []int
	for axis := 0; axis < operandShape.Rank(); axis++ {
		if !mappedSet[axis] {
			unmapped = append(unmapped, axis)
		}
	}

	operandPerm := append(startIndexMap, unmapped...)
	transposedOperandVal, err := f.Transpose(operandNode, operandPerm...)
	if err != nil {
		return nil, errors.Wrap(err, "Gather: failed to transpose operand to place mapped axes first")
	}
	transposedOperand := transposedOperandVal.(*Node)

	// 3. Perform GatherND on the transposed operand.
	K := indicesNode.shape.Rank() - 1
	gatherNDRank := K + len(unmapped)
	gatherNDDims := make([]int, gatherNDRank)
	copy(gatherNDDims, indicesNode.shape.Dimensions[:K])
	copy(gatherNDDims[K:], transposedOperand.shape.Dimensions[V:])
	gatherNDShape := shapes.Make(operandShape.DType, gatherNDDims...)

	f.nodeCount++
	gatherNDNode := &Node{
		name:   fmt.Sprintf("node_%d", f.nodeCount),
		opType: "GatherND",
		inputs: []*Node{transposedOperand, indicesNode},
		shape:  gatherNDShape,
		attributes: []*onnx.AttributeProto{
			{
				Name: "batch_dims",
				Type: onnx.AttributeProto_INT,
				I:    0,
			},
		},
	}
	f.nodes = append(f.nodes, gatherNDNode)

	// 4. Apply unmapped slicing if needed (if sliceSizes[u] < operandShape.Dimensions[u]).
	for idx, u := range unmapped {
		expectedSize := sliceSizes[u]
		actualSize := transposedOperand.shape.Dimensions[V+idx]
		if expectedSize < actualSize {
			start := make([]int, gatherNDRank)
			limit := make([]int, gatherNDRank)
			stride := make([]int, gatherNDRank)
			for i := 0; i < gatherNDRank; i++ {
				start[i] = 0
				limit[i] = gatherNDNode.shape.Dimensions[i]
				stride[i] = 1
			}
			limit[K+idx] = expectedSize
			slicedVal, err := f.Slice(gatherNDNode, start, limit, stride)
			if err != nil {
				return nil, errors.Wrap(err, "Gather: failed to slice unmapped dimension to match slice size")
			}
			gatherNDNode = slicedVal.(*Node)
		}
	}

	// 5. Reshape to B + O shape.
	// B are the batch dimensions of startIndices (excluding vector axis).
	// O are the non-collapsed slice dimensions.
	var O []int
	collapsedSet := make(map[int]bool)
	for _, axis := range collapsedSliceAxes {
		collapsedSet[axis] = true
	}
	for axis := 0; axis < operandShape.Rank(); axis++ {
		if !collapsedSet[axis] {
			O = append(O, sliceSizes[axis])
		}
	}

	B := indicesNode.shape.Dimensions[:K]
	reshapedDims := make([]int, len(B)+len(O))
	copy(reshapedDims, B)
	copy(reshapedDims[len(B):], O)
	reshapedVal, err := f.Reshape(gatherNDNode, reshapedDims...)
	if err != nil {
		return nil, errors.Wrap(err, "Gather: failed to reshape GatherND output to B+O shape")
	}
	reshapedNode := reshapedVal.(*Node)

	// 6. Transpose to the final output shape.
	finalRank := len(reshapedDims)
	P := make([]int, finalRank)
	for i := 0; i < finalRank; i++ {
		offsetIdx := -1
		for idx, axis := range offsetOutputAxes {
			if axis == i {
				offsetIdx = idx
				break
			}
		}
		if offsetIdx != -1 {
			P[i] = len(B) + offsetIdx
		} else {
			// Find the index of this batch dimension in B.
			j := 0
			for prev := 0; prev < i; prev++ {
				isOffset := false
				for _, axis := range offsetOutputAxes {
					if axis == prev {
						isOffset = true
						break
					}
				}
				if !isOffset {
					j++
				}
			}
			P[i] = j
		}
	}

	finalVal, err := f.Transpose(reshapedNode, P...)
	if err != nil {
		return nil, errors.Wrap(err, "Gather: failed to transpose reshaped output to final output axes placement")
	}
	return finalVal, nil
}
