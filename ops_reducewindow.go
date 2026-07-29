// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"fmt"

	"github.com/gomlx/compute"
	onnx "github.com/gomlx/compute-onnx/internal/protos"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func (f *Function) ReduceWindow(
	operand compute.Value,
	reductionType compute.ReduceOpType,
	windowDimensions, strides, inputDilations, windowDilations []int,
	paddings [][2]int,
) (compute.Value, error) {
	opNode, ok := operand.(*Node)
	if !ok {
		return nil, errors.New("operand must be a valid onnxruntime node")
	}

	outShape, err := shapeinference.ReduceWindow(opNode.shape, windowDimensions, strides, inputDilations, windowDilations, paddings)
	if err != nil {
		return nil, err
	}

	rank := opNode.shape.Rank()

	// Unsupported feature checks: ONNX pooling operators (MaxPool/AveragePool) do not support input dilations.
	for _, d := range inputDilations {
		if d > 1 {
			return nil, errors.Wrap(compute.ErrNotImplemented, "input dilations > 1 not supported by ONNX pooling operators")
		}
	}

	var ortOpType string
	switch reductionType {
	case compute.ReduceOpMax:
		ortOpType = "MaxPool"
	case compute.ReduceOpSum:
		ortOpType = "AveragePool"
	default:
		return nil, errors.Wrapf(compute.ErrNotImplemented, "ReduceWindow reduction type %s not implemented", reductionType)
	}

	// If operand rank < 3 (e.g. 1D vector), unsqueeze to 4D (1, 1, 1, L) for ONNX pooling.
	if rank < 3 {
		// Fill missing stride / windowDilation / padding defaults if nil
		effectiveWindow := windowDimensions
		effectiveStrides := strides
		if len(effectiveStrides) == 0 {
			effectiveStrides = effectiveWindow
		}
		effectiveWindowDilations := windowDilations
		if len(effectiveWindowDilations) == 0 {
			effectiveWindowDilations = make([]int, rank)
			for i := range effectiveWindowDilations {
				effectiveWindowDilations[i] = 1
			}
		}
		effectivePaddings := paddings
		if len(effectivePaddings) == 0 {
			effectivePaddings = make([][2]int, rank)
		}

		// Reshape 1D -> 4D: (1, 1, 1, N)
		reshapeDims := make([]int, 4)
		reshapeDims[0] = 1
		reshapeDims[1] = 1
		reshapeDims[2] = 1
		for i := 0; i < rank; i++ {
			reshapeDims[3-(rank-1)+i] = opNode.shape.Dimensions[i]
		}
		in4D, errR := f.Reshape(opNode, reshapeDims...)
		if errR != nil {
			return nil, errR
		}

		win4D := []int{1, 1, 1, effectiveWindow[0]}
		stride4D := []int{1, 1, 1, effectiveStrides[0]}
		winDil4D := []int{1, 1, 1, effectiveWindowDilations[0]}
		pad4D := [][2]int{{0, 0}, {0, 0}, {0, 0}, effectivePaddings[0]}

		res4D, errP := f.ReduceWindow(in4D, reductionType, win4D, stride4D, nil, winDil4D, pad4D)
		if errP != nil {
			return nil, errP
		}

		// Reshape back to outShape
		return f.Reshape(res4D, outShape.Dimensions...)
	}

	numSpatial := rank - 2

	// Determine channel axis from windowDimensions (must be 1 for batch axis 0, and 1 for channel axis).
	var channelAxis int
	if windowDimensions[0] == 1 && windowDimensions[1] == 1 {
		channelAxis = 1
	} else if windowDimensions[0] == 1 && windowDimensions[rank-1] == 1 {
		channelAxis = rank - 1
	} else {
		return nil, errors.Wrap(compute.ErrNotImplemented, "ReduceWindow requires batch and channel window dimensions to be 1")
	}

	inputPerm := make([]int, rank)
	inputPerm[0] = 0           // Batch
	inputPerm[1] = channelAxis // Channel
	spatialIdx := 2
	for i := 1; i < rank; i++ {
		if i != channelAxis {
			inputPerm[spatialIdx] = i
			spatialIdx++
		}
	}

	needInputTranspose := false
	for i, p := range inputPerm {
		if p != i {
			needInputTranspose = true
			break
		}
	}

	var inpVal compute.Value = opNode
	if needInputTranspose {
		inpVal, err = f.Transpose(opNode, inputPerm...)
		if err != nil {
			return nil, errors.Wrap(err, "ReduceWindow: input transpose failed")
		}
	}

	// 2. Prepare attributes for ONNX MaxPool / AveragePool (kernel_shape, pads, strides, dilations)
	kernelShape := make([]int64, numSpatial)
	for i := 0; i < numSpatial; i++ {
		origSpatialAxis := inputPerm[2+i]
		if len(windowDimensions) > origSpatialAxis {
			kernelShape[i] = int64(windowDimensions[origSpatialAxis])
		} else {
			kernelShape[i] = 1
		}
	}

	onnxPads := make([]int64, numSpatial*2)
	if len(paddings) > 0 {
		for i := 0; i < numSpatial; i++ {
			origSpatialAxis := inputPerm[2+i]
			if origSpatialAxis < len(paddings) {
				onnxPads[i] = int64(paddings[origSpatialAxis][0])            // begin
				onnxPads[i+numSpatial] = int64(paddings[origSpatialAxis][1]) // end
			}
		}
	}

	onnxStrides := make([]int64, numSpatial)
	for i := 0; i < numSpatial; i++ {
		origSpatialAxis := inputPerm[2+i]
		if len(strides) > origSpatialAxis {
			onnxStrides[i] = int64(strides[origSpatialAxis])
		} else {
			onnxStrides[i] = int64(windowDimensions[origSpatialAxis])
		}
	}

	onnxDilations := make([]int64, numSpatial)
	for i := 0; i < numSpatial; i++ {
		origSpatialAxis := inputPerm[2+i]
		if len(windowDilations) > origSpatialAxis {
			onnxDilations[i] = int64(windowDilations[origSpatialAxis])
		} else {
			onnxDilations[i] = 1
		}
	}

	attributes := []*onnx.AttributeProto{
		{
			Name: "kernel_shape",
			Type: onnx.AttributeProto_INTS,
			Ints: kernelShape,
		},
		{
			Name: "pads",
			Type: onnx.AttributeProto_INTS,
			Ints: onnxPads,
		},
		{
			Name: "strides",
			Type: onnx.AttributeProto_INTS,
			Ints: onnxStrides,
		},
	}

	if ortOpType == "MaxPool" {
		attributes = append(attributes, &onnx.AttributeProto{
			Name: "dilations",
			Type: onnx.AttributeProto_INTS,
			Ints: onnxDilations,
		})
	}

	// 3. Intermediate NCHW output shape for Pool node
	poolOutDims := make([]int, rank)
	poolOutDims[0] = outShape.Dimensions[0]
	poolOutDims[1] = outShape.Dimensions[channelAxis]
	for i := 0; i < numSpatial; i++ {
		origSpatialAxis := inputPerm[2+i]
		poolOutDims[2+i] = outShape.Dimensions[origSpatialAxis]
	}
	poolOutShape := shapes.Make(outShape.DType, poolOutDims...)

	f.nodeCount++
	poolNode := &Node{
		name:       fmt.Sprintf("node_%d", f.nodeCount),
		opType:     ortOpType,
		inputs:     []*Node{inpVal.(*Node)},
		shape:      poolOutShape,
		attributes: attributes,
	}
	f.nodes = append(f.nodes, poolNode)

	// If reductionType is ReduceOpSum, AveragePool computes the mean over window elements.
	// Multiply by the window element count to get the sum.
	var poolResult compute.Value = poolNode
	if reductionType == compute.ReduceOpSum {
		windowElements := 1
		for _, k := range kernelShape {
			windowElements *= int(k)
		}
		if windowElements > 1 {
			scaleConst, errC := f.Constant([]float32{float32(windowElements)})
			if errC != nil {
				return nil, errC
			}
			poolResult, err = f.Mul(poolNode, scaleConst)
			if err != nil {
				return nil, err
			}
		}
	}

	// 4. Transpose pool output back to target layout if needed
	outputPerm := make([]int, rank)
	outputPerm[0] = 0
	outputPerm[channelAxis] = 1
	for i := 0; i < numSpatial; i++ {
		origSpatialAxis := inputPerm[2+i]
		outputPerm[origSpatialAxis] = 2 + i
	}

	needOutputTranspose := false
	for i, p := range outputPerm {
		if p != i {
			needOutputTranspose = true
			break
		}
	}

	if needOutputTranspose {
		return f.Transpose(poolResult, outputPerm...)
	}

	return poolResult, nil
}
