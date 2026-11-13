// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"slices"

	"github.com/gomlx/compute"
	onnx "github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	registerOp(compute.OpTypeGather)
}

func makeShape(dtype dtypes.DType, dims []int, axisNames []string) shapes.Shape {
	isDynamic := slices.Contains(dims, shapes.DynamicDim)
	if isDynamic {
		newNames := make([]string, len(dims))
		if len(axisNames) == len(dims) {
			copy(newNames, axisNames)
		}
		for i, d := range dims {
			if d == shapes.DynamicDim && newNames[i] == "" {
				newNames[i] = shapes.AnonymousAxis
			}
		}
		return shapes.MakeDynamic(dtype, dims, newNames)
	}
	if len(axisNames) == len(dims) && slices.ContainsFunc(axisNames, func(s string) bool { return s != "" }) {
		return shapes.Make(dtype, dims...).WithAxisNames(axisNames...)
	}
	return shapes.Make(dtype, dims...)
}

func isIdentityPerm(perm []int) bool {
	for i, p := range perm {
		if p != i {
			return false
		}
	}
	return true
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
	if indexVectorAxis < 0 || indexVectorAxis > Q {
		return nil, errors.Errorf("Gather: indexVectorAxis=%d is out of range for startIndices %s", indexVectorAxis, startIndicesShape)
	}

	// 1. Prepare indices so the coordinate vector axis is the trailing axis [B..., V].
	var indicesNode *Node
	effStartIndicesShape := startIndicesShape
	effIndexVectorAxis := indexVectorAxis
	if indexVectorAxis == Q {
		// Virtual indexVectorAxis of size 1: append a trailing axis of size 1.
		if startIndicesShape.IsDynamic() {
			specs := make([]compute.DynamicDimensionSpec, Q+1)
			for i := range Q {
				if startIndicesShape.Dimensions[i] == shapes.DynamicDim {
					dimVal, err := f.DynamicDimensionSize(startIndicesNode, i)
					if err != nil {
						return nil, errors.Wrap(err, "Gather: failed to get dynamic dimension size for startIndices")
					}
					specs[i] = compute.DynamicDimensionSpec{
						Name:  startIndicesShape.AxisName(i),
						Value: dimVal,
					}
				} else {
					specs[i] = compute.DynamicDimensionSpec{Static: startIndicesShape.Dimensions[i]}
				}
			}
			specs[Q] = compute.DynamicDimensionSpec{Static: 1}
			indicesNodeVal, err := f.DynamicReshape(startIndicesNode, specs...)
			if err != nil {
				return nil, errors.Wrap(err, "Gather: failed to dynamic reshape startIndices to append virtual vector axis")
			}
			indicesNode = indicesNodeVal.(*Node)
		} else {
			newDims := make([]int, Q+1)
			copy(newDims, startIndicesShape.Dimensions)
			newDims[Q] = 1
			indicesNodeVal, err := f.Reshape(startIndicesNode, newDims...)
			if err != nil {
				return nil, errors.Wrap(err, "Gather: failed to reshape startIndices to append virtual vector axis")
			}
			indicesNode = indicesNodeVal.(*Node)
		}
		effStartIndicesShape = indicesNode.shape
		effIndexVectorAxis = Q
	} else if indexVectorAxis == Q-1 {
		indicesNode = startIndicesNode
	} else {
		perm := make([]int, Q)
		idx := 0
		for i := range Q {
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

	rawOutShape, err := shapeinference.Gather(
		operandShape,
		effStartIndicesShape,
		effIndexVectorAxis,
		offsetOutputAxes,
		collapsedSliceAxes,
		startIndexMap,
		sliceSizes,
		indicesAreSorted,
	)
	if err != nil {
		return nil, err
	}
	outShape := makeShape(rawOutShape.DType, rawOutShape.Dimensions, rawOutShape.AxisNames)

	collapsedSet := make(map[int]bool, len(collapsedSliceAxes))
	for _, axis := range collapsedSliceAxes {
		collapsedSet[axis] = true
	}

	// Fast path when any output dimension is statically zero (e.g. sliceSizes[axis] == 0 or empty batch/operand).
	if slices.Contains(outShape.Dimensions, 0) {
		if !outShape.IsDynamic() {
			return f.Constant(dtypes.MakeAnySlice(outShape.DType, 0), outShape.Dimensions...)
		}
		emptyConst, err := f.Constant(dtypes.MakeAnySlice(outShape.DType, 0), 0)
		if err != nil {
			return nil, err
		}
		var nonCollapsedOpAxes []int
		for axis := range operandShape.Rank() {
			if !collapsedSet[axis] {
				nonCollapsedOpAxes = append(nonCollapsedOpAxes, axis)
			}
		}
		specs := make([]compute.DynamicDimensionSpec, outShape.Rank())
		batchIdx := 0
		for outAxis := range outShape.Rank() {
			if offsetPos := slices.Index(offsetOutputAxes, outAxis); offsetPos != -1 {
				opAxis := nonCollapsedOpAxes[offsetPos]
				if outShape.Dimensions[outAxis] == shapes.DynamicDim {
					dimVal, err := f.DynamicDimensionSize(operandNode, opAxis)
					if err != nil {
						return nil, err
					}
					specs[outAxis] = compute.DynamicDimensionSpec{
						Name:  outShape.AxisName(outAxis),
						Value: dimVal,
					}
				} else {
					specs[outAxis] = compute.DynamicDimensionSpec{Static: outShape.Dimensions[outAxis]}
				}
			} else {
				if outShape.Dimensions[outAxis] == shapes.DynamicDim {
					dimVal, err := f.DynamicDimensionSize(indicesNode, batchIdx)
					if err != nil {
						return nil, err
					}
					specs[outAxis] = compute.DynamicDimensionSpec{
						Name:  outShape.AxisName(outAxis),
						Value: dimVal,
					}
				} else {
					specs[outAxis] = compute.DynamicDimensionSpec{Static: outShape.Dimensions[outAxis]}
				}
				batchIdx++
			}
		}
		return f.DynamicReshape(emptyConst, specs...)
	}

	for _, axis := range startIndexMap {
		if sliceSizes[axis] == shapes.DynamicDim {
			return nil, errors.Wrapf(compute.ErrNotImplemented, "Gather with dynamic sliceSize on mapped axis %d is not implemented for ONNX backend", axis)
		}
	}

	if indicesNode.shape.DType != dtypes.Int64 {
		castedIndices, err := f.ConvertDType(indicesNode, dtypes.Int64)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: failed to cast indices to Int64")
		}
		indicesNode = castedIndices.(*Node)
	}

	K := indicesNode.shape.Rank() - 1
	V := indicesNode.shape.Dimensions[K]

	// 2. Clamp indices to [0, max(0, dim - sliceSizes[axis])] to match XLA / StableHLO semantics.
	if V > 0 {
		zeroConst, err := f.MakeScalar(int64(0), dtypes.Int64)
		if err != nil {
			return nil, err
		}
		allMappedStatic := true
		for _, axis := range startIndexMap {
			if operandShape.Dimensions[axis] == shapes.DynamicDim {
				allMappedStatic = false
				break
			}
		}

		var maxBoundVal compute.Value
		if allMappedStatic {
			maxBounds := make([]int64, V)
			for i, axis := range startIndexMap {
				maxBounds[i] = int64(max(0, operandShape.Dimensions[axis]-sliceSizes[axis]))
			}
			boundDims := make([]int, K+1)
			for i := range K {
				boundDims[i] = 1
			}
			boundDims[K] = V
			maxBoundVal, err = f.Constant(maxBounds, boundDims...)
			if err != nil {
				return nil, err
			}
		} else {
			zero1D, err := f.Constant([]int64{0}, 1)
			if err != nil {
				return nil, err
			}
			boundParts := make([]compute.Value, V)
			for i, axis := range startIndexMap {
				dim := operandShape.Dimensions[axis]
				if dim != shapes.DynamicDim {
					c, err := f.Constant([]int64{int64(max(0, dim-sliceSizes[axis]))}, 1)
					if err != nil {
						return nil, err
					}
					boundParts[i] = c
				} else {
					dimScalar, err := f.DynamicDimensionSize(operandNode, axis)
					if err != nil {
						return nil, err
					}
					dim1D, err := f.Reshape(dimScalar, 1)
					if err != nil {
						return nil, err
					}
					sliceConst, err := f.Constant([]int64{int64(sliceSizes[axis])}, 1)
					if err != nil {
						return nil, err
					}
					subVal, err := f.Sub(dim1D, sliceConst)
					if err != nil {
						return nil, err
					}
					clampedBound, err := f.Max(subVal, zero1D)
					if err != nil {
						return nil, err
					}
					boundParts[i] = clampedBound
				}
			}
			var bound1D compute.Value
			if V == 1 {
				bound1D = boundParts[0]
			} else {
				bound1D, err = f.Concatenate(0, boundParts...)
				if err != nil {
					return nil, err
				}
			}
			if K == 0 {
				maxBoundVal = bound1D
			} else {
				boundDims := make([]int, K+1)
				for i := range K {
					boundDims[i] = 1
				}
				boundDims[K] = V
				maxBoundVal, err = f.Reshape(bound1D, boundDims...)
				if err != nil {
					return nil, err
				}
			}
		}

		clampedMin, err := f.Max(indicesNode, zeroConst)
		if err != nil {
			return nil, err
		}
		clampedMax, err := f.Min(clampedMin, maxBoundVal)
		if err != nil {
			return nil, err
		}
		indicesNode = clampedMax.(*Node)
	}

	// 3. Expand non-collapsed mapped axes into explicit window coordinate grids [B..., W_0..W_{M-1}, V]
	// so a single GatherND gathers arbitrary window sizes without recursive Concatenate calls.
	var windowComps []int
	var windowSizes []int
	var mappedSliceOpAxes []int
	for comp, opAxis := range startIndexMap {
		if !collapsedSet[opAxis] {
			windowComps = append(windowComps, comp)
			windowSizes = append(windowSizes, sliceSizes[opAxis])
			mappedSliceOpAxes = append(mappedSliceOpAxes, opAxis)
		}
	}
	M := len(windowComps)
	gatherIndicesNode := indicesNode
	if M > 0 {
		// Reshape indicesNode from [B..., V] to [B..., 1..1 (M ones), V].
		if indicesNode.shape.IsDynamic() {
			specs := make([]compute.DynamicDimensionSpec, K+M+1)
			for i := range K {
				if indicesNode.shape.Dimensions[i] == shapes.DynamicDim {
					dimVal, err := f.DynamicDimensionSize(indicesNode, i)
					if err != nil {
						return nil, err
					}
					specs[i] = compute.DynamicDimensionSpec{
						Name:  indicesNode.shape.AxisName(i),
						Value: dimVal,
					}
				} else {
					specs[i] = compute.DynamicDimensionSpec{Static: indicesNode.shape.Dimensions[i]}
				}
			}
			for m := range M {
				specs[K+m] = compute.DynamicDimensionSpec{Static: 1}
			}
			specs[K+M] = compute.DynamicDimensionSpec{Static: V}
			reshapedIdx, err := f.DynamicReshape(indicesNode, specs...)
			if err != nil {
				return nil, errors.Wrap(err, "Gather: failed to dynamic reshape indices for window expansion")
			}
			gatherIndicesNode = reshapedIdx.(*Node)
		} else {
			newDims := make([]int, K+M+1)
			copy(newDims, indicesNode.shape.Dimensions[:K])
			for m := range M {
				newDims[K+m] = 1
			}
			newDims[K+M] = V
			reshapedIdx, err := f.Reshape(indicesNode, newDims...)
			if err != nil {
				return nil, errors.Wrap(err, "Gather: failed to reshape indices for window expansion")
			}
			gatherIndicesNode = reshapedIdx.(*Node)
		}

		hasWindowOffset := slices.ContainsFunc(windowSizes, func(w int) bool { return w > 1 })
		if hasWindowOffset {
			numPoints := 1
			for _, w := range windowSizes {
				numPoints *= w
			}
			offsets := make([]int64, numPoints*V)
			stridesGrid := make([]int, M)
			strideAcc := 1
			for m := M - 1; m >= 0; m-- {
				stridesGrid[m] = strideAcc
				strideAcc *= windowSizes[m]
			}
			for pt := range numPoints {
				base := pt * V
				for m, comp := range windowComps {
					coord := (pt / stridesGrid[m]) % windowSizes[m]
					offsets[base+comp] = int64(coord)
				}
			}
			gridDims := make([]int, K+M+1)
			for i := range K {
				gridDims[i] = 1
			}
			copy(gridDims[K:], windowSizes)
			gridDims[K+M] = V
			gridConst, err := f.Constant(offsets, gridDims...)
			if err != nil {
				return nil, err
			}
			addedIdx, err := f.Add(gatherIndicesNode, gridConst)
			if err != nil {
				return nil, errors.Wrap(err, "Gather: failed to add window offsets to indices")
			}
			gatherIndicesNode = addedIdx.(*Node)
		}
	}

	// 4. Prepare operand by transposing the mapped dimensions to the front.
	mappedSet := make(map[int]bool, len(startIndexMap))
	for _, axis := range startIndexMap {
		mappedSet[axis] = true
	}
	var unmapped []int
	for axis := range operandShape.Rank() {
		if !mappedSet[axis] {
			unmapped = append(unmapped, axis)
		}
	}

	operandPerm := slices.Concat(startIndexMap, unmapped)
	transposedOperand := operandNode
	if !isIdentityPerm(operandPerm) {
		transposedOperandVal, err := f.Transpose(operandNode, operandPerm...)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: failed to transpose operand to place mapped axes first")
		}
		transposedOperand = transposedOperandVal.(*Node)
	}

	// 5. Perform GatherND on the transposed operand.
	gatherBatchRank := K + M
	gatherNDRank := gatherBatchRank + len(unmapped)
	gatherNDDims := make([]int, gatherNDRank)
	copy(gatherNDDims, gatherIndicesNode.shape.Dimensions[:gatherBatchRank])
	copy(gatherNDDims[gatherBatchRank:], transposedOperand.shape.Dimensions[V:])
	axisNames := make([]string, gatherNDRank)
	if gatherIndicesNode.shape.AxisNames != nil {
		copy(axisNames, gatherIndicesNode.shape.AxisNames[:gatherBatchRank])
	}
	if transposedOperand.shape.AxisNames != nil {
		copy(axisNames[gatherBatchRank:], transposedOperand.shape.AxisNames[V:])
	}
	gatherNDShape := makeShape(operandShape.DType, gatherNDDims, axisNames)

	gatherNDNode := f.addNode(&Node{
		opType: "GatherND",
		inputs: []*Node{transposedOperand, gatherIndicesNode},
		shape:  gatherNDShape,
		attributes: []*onnx.AttributeProto{
			{
				Name: "batch_dims",
				Type: onnx.AttributeProto_INT,
				I:    0,
			},
		},
	})

	// 6. Apply unmapped slicing if needed (when sliceSizes[u] is static and smaller than or narrowing a dynamic unmapped axis).
	for idx, u := range unmapped {
		expectedSize := sliceSizes[u]
		actualSize := transposedOperand.shape.Dimensions[V+idx]
		if expectedSize != shapes.DynamicDim && (actualSize == shapes.DynamicDim || expectedSize < actualSize) {
			start := make([]int, gatherNDRank)
			limit := make([]int, gatherNDRank)
			stride := make([]int, gatherNDRank)
			for i := range gatherNDRank {
				start[i] = 0
				limit[i] = gatherNDNode.shape.Dimensions[i]
				stride[i] = 1
			}
			limit[gatherBatchRank+idx] = expectedSize
			slicedVal, err := f.Slice(gatherNDNode, start, limit, stride)
			if err != nil {
				return nil, errors.Wrap(err, "Gather: failed to slice unmapped dimension to match slice size")
			}
			gatherNDNode = slicedVal.(*Node)
		}
	}

	// 7. Squeeze any collapsed unmapped axes (each has static sliceSize == 1).
	var unmappedNonCollapsedAxes []int
	var keepGatherNDAxes []int
	for i := range gatherBatchRank {
		keepGatherNDAxes = append(keepGatherNDAxes, i)
	}
	hasCollapsedUnmapped := false
	for idx, u := range unmapped {
		if collapsedSet[u] {
			hasCollapsedUnmapped = true
		} else {
			unmappedNonCollapsedAxes = append(unmappedNonCollapsedAxes, u)
			keepGatherNDAxes = append(keepGatherNDAxes, gatherBatchRank+idx)
		}
	}

	slicedNode := gatherNDNode
	if hasCollapsedUnmapped {
		if gatherNDNode.shape.IsDynamic() {
			specs := make([]compute.DynamicDimensionSpec, len(keepGatherNDAxes))
			for i, srcAxis := range keepGatherNDAxes {
				dim := gatherNDNode.shape.Dimensions[srcAxis]
				if dim == shapes.DynamicDim {
					dimVal, err := f.DynamicDimensionSize(gatherNDNode, srcAxis)
					if err != nil {
						return nil, errors.Wrap(err, "Gather: failed to get dynamic dimension size while squeezing collapsed axes")
					}
					specs[i] = compute.DynamicDimensionSpec{
						Name:  gatherNDNode.shape.AxisName(srcAxis),
						Value: dimVal,
					}
				} else {
					specs[i] = compute.DynamicDimensionSpec{Static: dim}
				}
			}
			reshapedVal, err := f.DynamicReshape(gatherNDNode, specs...)
			if err != nil {
				return nil, errors.Wrap(err, "Gather: failed to dynamic reshape GatherND output")
			}
			slicedNode = reshapedVal.(*Node)
		} else {
			newDims := make([]int, len(keepGatherNDAxes))
			for i, srcAxis := range keepGatherNDAxes {
				newDims[i] = gatherNDNode.shape.Dimensions[srcAxis]
			}
			reshapedVal, err := f.Reshape(gatherNDNode, newDims...)
			if err != nil {
				return nil, errors.Wrap(err, "Gather: failed to reshape GatherND output")
			}
			slicedNode = reshapedVal.(*Node)
		}
	}

	// 8. Permute [B..., mappedSliceOpAxes..., unmappedNonCollapsedAxes...] directly to final output layout.
	sliceOpAxes := slices.Concat(mappedSliceOpAxes, unmappedNonCollapsedAxes)
	var sortedNonCollapsedOpAxes []int
	for axis := range operandShape.Rank() {
		if !collapsedSet[axis] {
			sortedNonCollapsedOpAxes = append(sortedNonCollapsedOpAxes, axis)
		}
	}

	finalRank := K + len(sliceOpAxes)
	P := make([]int, finalRank)
	batchIdx := 0
	for outAxis := range finalRank {
		if offsetPos := slices.Index(offsetOutputAxes, outAxis); offsetPos != -1 {
			targetOpAxis := sortedNonCollapsedOpAxes[offsetPos]
			posInSliced := slices.Index(sliceOpAxes, targetOpAxis)
			P[outAxis] = K + posInSliced
		} else {
			P[outAxis] = batchIdx
			batchIdx++
		}
	}

	finalNode := slicedNode
	if !isIdentityPerm(P) {
		finalVal, err := f.Transpose(slicedNode, P...)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: failed to transpose gathered output to final output axes placement")
		}
		finalNode = finalVal.(*Node)
	}
	finalNode.shape = outShape
	return finalNode, nil
}
