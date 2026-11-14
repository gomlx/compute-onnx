// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"slices"
	"strings"

	"github.com/gomlx/compute"
	onnx "github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func (f *Function) ConvGeneral(input, kernel compute.Value, axes compute.ConvolveAxesConfig, strides []int, paddings [][2]int, inputDilations []int, kernelDilations []int, channelGroupCount int, batchGroupCount int) (compute.Value, error) {
	if channelGroupCount == 0 {
		channelGroupCount = 1
	}
	if batchGroupCount == 0 {
		batchGroupCount = 1
	}

	inputNode, ok1 := input.(*Node)
	kernelNode, ok2 := kernel.(*Node)
	if !ok1 || !ok2 {
		return nil, errors.New("inputs must be valid onnxruntime nodes")
	}

	rawOutShape, err := shapeinference.ConvGeneral(inputNode.shape, kernelNode.shape, axes, strides, paddings, inputDilations, kernelDilations, channelGroupCount, batchGroupCount)
	if err != nil {
		return nil, err
	}

	rank := inputNode.shape.Rank()
	numSpatial := rank - 2

	// ONNX Conv operator does not support BFloat16 in standard ONNX Runtime schemas.
	if inputNode.shape.DType == dtypes.BFloat16 {
		return nil, errors.Wrapf(compute.ErrNotImplemented, "ONNX doesn't support BFloat16 for Conv: standard ONNX Conv schema does not support tensor(bfloat16)")
	}
	// ONNX Conv operator does not natively support batch grouping (batchGroupCount > 1).
	if batchGroupCount > 1 {
		return nil, errors.Wrap(compute.ErrNotImplemented, "batchGroupCount > 1 not supported by ONNX Conv")
	}

	// An empty static output has no convolution windows to evaluate.
	if !rawOutShape.IsDynamic() && rawOutShape.Size() == 0 {
		return f.Constant(dtypes.MakeAnySlice(rawOutShape.DType, 0), rawOutShape.Dimensions...)
	}

	// Lower input dilations (atrous/upsampled inputs, e.g. from transposed convolutions) and/or negative
	// paddings (edge cropping) via Pad / DynamicPad before delegating to standard ONNX Conv.
	hasInputDilation := slices.ContainsFunc(inputDilations, func(d int) bool { return d > 1 })
	hasNegativePadding := slices.ContainsFunc(paddings, func(p [2]int) bool { return p[0] < 0 || p[1] < 0 })
	if hasInputDilation || hasNegativePadding {
		zero, err := f.MakeScalar(0, inputNode.shape.DType)
		if err != nil {
			return nil, err
		}
		remPaddings := make([][2]int, numSpatial)
		copy(remPaddings, paddings)

		dilatedDynamicSpatial := false
		for i, spatialAxis := range axes.InputSpatial {
			d := 1
			if i < len(inputDilations) {
				d = inputDilations[i]
			}
			var p [2]int
			if i < len(paddings) {
				p = paddings[i]
			}
			if (d > 1 || p[0] < 0 || p[1] < 0) && inputNode.shape.Dimensions[spatialAxis] == shapes.DynamicDim {
				dilatedDynamicSpatial = true
			}
		}

		var padded compute.Value
		if dilatedDynamicSpatial {
			dynPadConfig := make([]compute.DynamicPadAxis, rank)
			for i, spatialAxis := range axes.InputSpatial {
				d := 1
				if i < len(inputDilations) {
					d = inputDilations[i]
				}
				if d > 1 {
					dynPadConfig[spatialAxis].Interior = d - 1
				}
				if i < len(paddings) && (paddings[i][0] < 0 || paddings[i][1] < 0) {
					dynPadConfig[spatialAxis].Start = paddings[i][0]
					dynPadConfig[spatialAxis].End = paddings[i][1]
					remPaddings[i] = [2]int{0, 0}
				}
				if (d > 1 || dynPadConfig[spatialAxis].Start < 0 || dynPadConfig[spatialAxis].End < 0) &&
					inputNode.shape.Dimensions[spatialAxis] == shapes.DynamicDim {
					dynPadConfig[spatialAxis].TargetAxisName = shapes.AnonymousAxis
				}
			}
			padded, err = f.DynamicPad(inputNode, zero, dynPadConfig...)
		} else {
			padConfig := make([]compute.PadAxis, rank)
			for i, spatialAxis := range axes.InputSpatial {
				if i < len(inputDilations) && inputDilations[i] > 1 {
					padConfig[spatialAxis].Interior = inputDilations[i] - 1
				}
				if i < len(paddings) && (paddings[i][0] < 0 || paddings[i][1] < 0) {
					padConfig[spatialAxis].Start = paddings[i][0]
					padConfig[spatialAxis].End = paddings[i][1]
					remPaddings[i] = [2]int{0, 0}
				}
			}
			padded, err = f.Pad(inputNode, zero, padConfig...)
		}
		if err != nil {
			return nil, err
		}

		res, err := f.ConvGeneral(padded, kernelNode, axes, strides, remPaddings, nil, kernelDilations, channelGroupCount, batchGroupCount)
		if err != nil {
			return nil, err
		}
		resNode := res.(*Node)
		if rawOutShape.AxisNames != nil && resNode.shape.AxisNames != nil {
			updated := false
			known := f.KnownDynamicAxisNames()
			for i, rawName := range rawOutShape.AxisNames {
				if rawName == shapes.AnonymousAxis && !strings.HasPrefix(resNode.shape.AxisNames[i], "#conv_") {
					for {
						cand := f.UniqueAxisName("#conv")
						if !known.Has(cand) {
							resNode.shape.AxisNames[i] = cand
							known.Insert(cand)
							updated = true
							break
						}
					}
				}
			}
			if updated {
				f.recordDynamicAxisSources(resNode)
			}
		}
		return resNode, nil
	}

	outShape := f.NameAnonymousDynamicAxes(rawOutShape, "#conv")

	// 1. Transpose input to NCHW / NCDHW format (Batch, Channels, Spatial...)
	inputPerm := make([]int, rank)
	inputPerm[0] = axes.InputBatch
	inputPerm[1] = axes.InputChannels
	copy(inputPerm[2:], axes.InputSpatial)

	needInputTranspose := false
	for i, p := range inputPerm {
		if p != i {
			needInputTranspose = true
			break
		}
	}

	var inpVal compute.Value = inputNode
	if needInputTranspose {
		inpVal, err = f.Transpose(inputNode, inputPerm...)
		if err != nil {
			return nil, errors.Wrap(err, "ConvGeneral: input transpose failed")
		}
	}

	// 2. Transpose kernel to OIHW / OIDHW format (OutputChannels, InputChannels, Spatial...)
	kernelPerm := make([]int, rank)
	kernelPerm[0] = axes.KernelOutputChannels
	kernelPerm[1] = axes.KernelInputChannels
	copy(kernelPerm[2:], axes.KernelSpatial)

	needKernelTranspose := false
	for i, p := range kernelPerm {
		if p != i {
			needKernelTranspose = true
			break
		}
	}

	var kerVal compute.Value = kernelNode
	if needKernelTranspose {
		kerVal, err = f.Transpose(kernelNode, kernelPerm...)
		if err != nil {
			return nil, errors.Wrap(err, "ConvGeneral: kernel transpose failed")
		}
	}

	// 3. Prepare Attributes for ONNX Conv (pads, strides, dilations, group)
	onnxPads := make([]int64, numSpatial*2)
	if len(paddings) > 0 {
		for i := range numSpatial {
			if i < len(paddings) {
				onnxPads[i] = int64(paddings[i][0])            // begin
				onnxPads[i+numSpatial] = int64(paddings[i][1]) // end
			}
		}
	}

	onnxStrides := make([]int64, numSpatial)
	for i := range numSpatial {
		if i < len(strides) {
			onnxStrides[i] = int64(strides[i])
		} else {
			onnxStrides[i] = 1
		}
	}

	onnxDilations := make([]int64, numSpatial)
	for i := range numSpatial {
		if i < len(kernelDilations) {
			onnxDilations[i] = int64(kernelDilations[i])
		} else {
			onnxDilations[i] = 1
		}
	}

	attributes := []*onnx.AttributeProto{
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
		{
			Name: "dilations",
			Type: onnx.AttributeProto_INTS,
			Ints: onnxDilations,
		},
	}

	if channelGroupCount > 1 {
		attributes = append(attributes, &onnx.AttributeProto{
			Name: "group",
			Type: onnx.AttributeProto_INT,
			I:    int64(channelGroupCount),
		})
	}

	// ONNX Conv operator in ONNX Runtime computes Conv outputs using Float32 buffers.
	// Cast integer and Float64 inputs to Float32 before Conv execution and cast output back.
	origDType := outShape.DType
	needCastToFloat32 := !origDType.IsFloat() || origDType == dtypes.Float64
	if needCastToFloat32 {
		var errCast error
		inpVal, errCast = f.ConvertDType(inpVal, dtypes.Float32)
		if errCast != nil {
			return nil, errCast
		}
		kerVal, errCast = f.ConvertDType(kerVal, dtypes.Float32)
		if errCast != nil {
			return nil, errCast
		}
	}

	// 4. Construct intermediate NCHW output shape for Conv node
	convOutDType := outShape.DType
	if needCastToFloat32 {
		convOutDType = dtypes.Float32
	}
	convOutDims := make([]int, rank)
	convOutDims[0] = outShape.Dimensions[axes.OutputBatch]
	convOutDims[1] = outShape.Dimensions[axes.OutputChannels]
	for i, spatialIdx := range axes.OutputSpatial {
		convOutDims[2+i] = outShape.Dimensions[spatialIdx]
	}
	var convOutAxisNames []string
	if outShape.AxisNames != nil {
		convOutAxisNames = make([]string, rank)
		convOutAxisNames[0] = outShape.AxisName(axes.OutputBatch)
		convOutAxisNames[1] = outShape.AxisName(axes.OutputChannels)
		for i, spatialIdx := range axes.OutputSpatial {
			convOutAxisNames[2+i] = outShape.AxisName(spatialIdx)
		}
	}
	convOutShape := makeShape(convOutDType, convOutDims, convOutAxisNames)

	convNode := f.addNode(&Node{
		opType:     "Conv",
		inputs:     []*Node{inpVal.(*Node), kerVal.(*Node)},
		shape:      convOutShape,
		attributes: attributes,
	})

	// 5. Transpose conv output back to target layout (OutputBatch, OutputChannels, OutputSpatial...)
	// convNode layout is [0=Batch, 1=Channels, 2..2+spatialRank-1=Spatial...]
	outputPerm := make([]int, rank)
	outputPerm[axes.OutputBatch] = 0
	outputPerm[axes.OutputChannels] = 1
	for i, spatialIdx := range axes.OutputSpatial {
		outputPerm[spatialIdx] = 2 + i
	}

	needOutputTranspose := false
	for i, p := range outputPerm {
		if p != i {
			needOutputTranspose = true
			break
		}
	}

	var finalVal compute.Value = convNode
	if needOutputTranspose {
		var errT error
		finalVal, errT = f.Transpose(convNode, outputPerm...)
		if errT != nil {
			return nil, errT
		}
	}

	if needCastToFloat32 {
		return f.ConvertDType(finalVal, origDType)
	}

	return finalVal, nil
}
