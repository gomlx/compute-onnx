// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"math"

	"github.com/gomlx/compute"
	onnx "github.com/gomlx/compute-onnx/internal/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/pkg/errors"
)

func init() {
	registerOp(compute.OpTypeFusedSoftmax)
	registerOp(compute.OpTypeFusedGelu)
	registerOp(compute.OpTypeFusedLayerNorm)
	registerOp(compute.OpTypeFusedDense)
	registerOp(compute.OpTypeFusedScaledDotProductAttention)
	registerOp(compute.OpTypeFusedAttentionQKVProjection)
	registerOp(compute.OpTypeFusedQuantizedDense)
}

func expandRankToMatch(f *Function, val compute.Value, targetRank int) (compute.Value, error) {
	valNode, ok := val.(*Node)
	if !ok {
		return nil, errors.New("expandRankToMatch: val must be a valid onnxruntime node")
	}
	currentRank := valNode.shape.Rank()
	if currentRank >= targetRank {
		return valNode, nil
	}
	diff := targetRank - currentRank
	newDims := make([]int, targetRank)
	for i := 0; i < diff; i++ {
		newDims[i] = 1
	}
	for i := 0; i < currentRank; i++ {
		newDims[diff+i] = valNode.shape.Dimensions[i]
	}
	return f.Reshape(valNode, newDims...)
}

// FusedSoftmax computes softmax along the specified axis using ONNX Softmax.
func (f *Function) FusedSoftmax(x compute.Value, axis int) (compute.Value, error) {
	xNode, ok := x.(*Node)
	if !ok {
		return nil, errors.New("FusedSoftmax: input must be a valid onnxruntime node")
	}
	rank := xNode.shape.Rank()
	if axis < 0 || axis >= rank {
		return nil, errors.Errorf("FusedSoftmax: axis %d out of bounds for shape rank %d", axis, rank)
	}

	node := &Node{
		opType: "Softmax",
		inputs: []*Node{xNode},
		shape:  xNode.shape,
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

// FusedGelu computes Gaussian Error Linear Unit activation using ONNX Gelu (opset 20+).
func (f *Function) FusedGelu(x compute.Value, exact bool) (compute.Value, error) {
	xNode, ok := x.(*Node)
	if !ok {
		return nil, errors.New("FusedGelu: input must be a valid onnxruntime node")
	}

	approxStr := "none"
	if !exact {
		approxStr = "tanh"
	}

	node := &Node{
		opType: "Gelu",
		inputs: []*Node{xNode},
		shape:  xNode.shape,
		attributes: []*onnx.AttributeProto{
			{
				Name: "approximate",
				Type: onnx.AttributeProto_STRING,
				S:    []byte(approxStr),
			},
		},
	}
	return f.addNode(node), nil
}

// FusedLayerNorm applies layer normalization over specified axes using ONNX LayerNormalization (opset 17+).
func (f *Function) FusedLayerNorm(x compute.Value, axes []int, epsilon float64, gamma, beta compute.Value) (compute.Value, error) {
	xNode, ok := x.(*Node)
	if !ok {
		return nil, errors.New("FusedLayerNorm: input must be a valid onnxruntime node")
	}

	rank := xNode.shape.Rank()
	if len(axes) == 0 || len(axes) > rank {
		return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedLayerNorm: invalid axes %v for rank %d", axes, rank)
	}

	// ONNX LayerNormalization requires axes to form a contiguous trailing suffix: [rank-k, ..., rank-1]
	k := len(axes)
	startAxis := rank - k
	for i, a := range axes {
		normalizedAxis := a
		if normalizedAxis < 0 {
			normalizedAxis += rank
		}
		if normalizedAxis != startAxis+i {
			return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedLayerNorm: ONNX LayerNormalization requires contiguous trailing axes [axis..rank-1], got axes=%v for rank %d", axes, rank)
		}
	}

	var gammaNode *Node
	if gamma != nil {
		var ok bool
		gammaNode, ok = gamma.(*Node)
		if !ok {
			return nil, errors.New("FusedLayerNorm: gamma must be a valid onnxruntime node")
		}
	} else {
		// Construct gamma as constant tensor of 1s matching normalized shape
		normDims := xNode.shape.Dimensions[startAxis:]
		size := 1
		for _, d := range normDims {
			size *= d
		}

		var constVal compute.Value
		var err error
		switch xNode.shape.DType {
		case dtypes.Float32:
			ones := make([]float32, size)
			for i := range ones {
				ones[i] = 1.0
			}
			constVal, err = f.Constant(ones, normDims...)
		case dtypes.Float64:
			ones := make([]float64, size)
			for i := range ones {
				ones[i] = 1.0
			}
			constVal, err = f.Constant(ones, normDims...)
		case dtypes.Float16:
			ones := make([]float16.Float16, size)
			for i := range ones {
				ones[i] = float16.FromFloat32(1.0)
			}
			constVal, err = f.Constant(ones, normDims...)
		case dtypes.BFloat16:
			ones := make([]bfloat16.BFloat16, size)
			for i := range ones {
				ones[i] = bfloat16.FromFloat32(1.0)
			}
			constVal, err = f.Constant(ones, normDims...)
		default:
			return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedLayerNorm: unsupported dtype %s for nil gamma", xNode.shape.DType)
		}
		if err != nil {
			return nil, err
		}
		gammaNode = constVal.(*Node)
	}

	inputs := []*Node{xNode, gammaNode}

	if beta != nil {
		betaNode, ok := beta.(*Node)
		if !ok {
			return nil, errors.New("FusedLayerNorm: beta must be a valid onnxruntime node")
		}
		inputs = append(inputs, betaNode)
	}

	node := &Node{
		opType: "LayerNormalization",
		inputs: inputs,
		shape:  xNode.shape,
		attributes: []*onnx.AttributeProto{
			{
				Name: "axis",
				Type: onnx.AttributeProto_INT,
				I:    int64(startAxis),
			},
			{
				Name: "epsilon",
				Type: onnx.AttributeProto_FLOAT,
				F:    float32(epsilon),
			},
		},
	}
	return f.addNode(node), nil
}

// FusedDense performs fused matmul + optional bias + optional activation.
func (f *Function) FusedDense(x, weight, bias compute.Value, options compute.DenseConfig) (compute.Value, error) {
	xNode, ok1 := x.(*Node)
	wNode, ok2 := weight.(*Node)
	if !ok1 || !ok2 {
		return nil, errors.New("FusedDense: inputs must be valid onnxruntime nodes")
	}

	xRank := xNode.shape.Rank()
	wRank := wNode.shape.Rank()
	if xRank < 1 || wRank < 1 {
		return nil, errors.Errorf("FusedDense: x rank (%d) and weight rank (%d) must be at least 1", xRank, wRank)
	}

	var lhsContractingAxes []int
	var rhsContractingAxes []int

	if options.WeightLayout == compute.DenseLayoutOutputsInput {
		lhsContractingAxes = []int{xRank - 1}
		rhsContractingAxes = []int{wRank - 1}
	} else {
		lhsContractingAxes = []int{xRank - 1}
		rhsContractingAxes = []int{0}
	}

	// Compute MatMul / DotGeneral
	dotRes, err := f.DotGeneral(xNode, lhsContractingAxes, nil, wNode, rhsContractingAxes, nil, compute.DotGeneralConfig{})
	if err != nil {
		return nil, errors.Wrap(err, "FusedDense: matmul failed")
	}

	// Add bias if provided
	outVal := dotRes
	if bias != nil {
		biasNode, ok := bias.(*Node)
		if !ok {
			return nil, errors.New("FusedDense: bias must be a valid onnxruntime node")
		}
		dotNode := dotRes.(*Node)
		biasReshaped, err := expandRankToMatch(f, biasNode, dotNode.shape.Rank())
		if err != nil {
			return nil, errors.Wrap(err, "FusedDense: bias reshape failed")
		}
		outVal, err = f.Add(dotRes, biasReshaped)
		if err != nil {
			return nil, errors.Wrap(err, "FusedDense: bias addition failed")
		}
	}

	// Apply activation
	switch options.Activation {
	case compute.ActivationNone:
		return outVal, nil
	case compute.ActivationRelu:
		outNode, ok := outVal.(*Node)
		if !ok {
			return nil, errors.New("FusedDense: outVal is not a valid onnxruntime node")
		}
		node := &Node{
			opType: "Relu",
			inputs: []*Node{outNode},
			shape:  outNode.shape,
		}
		return f.addNode(node), nil
	case compute.ActivationGelu:
		return f.FusedGelu(outVal, true)
	case compute.ActivationSilu:
		sig, err := f.Logistic(outVal)
		if err != nil {
			return nil, err
		}
		return f.Mul(outVal, sig)
	case compute.ActivationHardSwish:
		outNode, ok := outVal.(*Node)
		if !ok {
			return nil, errors.New("FusedDense: outVal is not a valid onnxruntime node")
		}
		node := &Node{
			opType: "HardSwish",
			inputs: []*Node{outNode},
			shape:  outNode.shape,
		}
		return f.addNode(node), nil
	case compute.ActivationTanh:
		return f.Tanh(outVal)
	default:
		return nil, errors.Errorf("FusedDense: unsupported activation type %v", options.Activation)
	}
}

// FusedScaledDotProductAttention computes multi-head scaled dot-product attention.
func (f *Function) FusedScaledDotProductAttention(
	query, key, value compute.Value,
	axesLayout compute.AttentionAxesLayout,
	options *compute.ScaledDotProductAttentionConfig) (output compute.Value, statesForVJP []compute.Value, err error) {

	qNode, ok1 := query.(*Node)
	kNode, ok2 := key.(*Node)
	vNode, ok3 := value.(*Node)
	if !ok1 || !ok2 || !ok3 {
		return nil, nil, errors.New("FusedScaledDotProductAttention: inputs must be valid onnxruntime nodes")
	}

	if qNode.shape.Rank() != 4 || kNode.shape.Rank() != 4 || vNode.shape.Rank() != 4 {
		return nil, nil, errors.Errorf("FusedScaledDotProductAttention: query, key, value must be 4D tensors, got ranks %d, %d, %d",
			qNode.shape.Rank(), kNode.shape.Rank(), vNode.shape.Rank())
	}

	if options != nil && (options.QuerySeqLen != nil || options.KeyValueSeqLen != nil) {
		return nil, nil, errors.Wrap(compute.ErrNotImplemented, "FusedScaledDotProductAttention: QuerySeqLen/KeyValueSeqLen not supported")
	}

	var qBHSD, kBHSD, vBHSD *Node
	if axesLayout == compute.AttentionAxesLayoutBSHD {
		qBHSDVal, err := f.Transpose(qNode, 0, 2, 1, 3)
		if err != nil {
			return nil, nil, err
		}
		kBHSDVal, err := f.Transpose(kNode, 0, 2, 1, 3)
		if err != nil {
			return nil, nil, err
		}
		vBHSDVal, err := f.Transpose(vNode, 0, 2, 1, 3)
		if err != nil {
			return nil, nil, err
		}
		qBHSD = qBHSDVal.(*Node)
		kBHSD = kBHSDVal.(*Node)
		vBHSD = vBHSDVal.(*Node)
	} else {
		qBHSD = qNode
		kBHSD = kNode
		vBHSD = vNode
	}

	numHeadsQ := qBHSD.shape.Dimensions[1]
	numHeadsKV := kBHSD.shape.Dimensions[1]
	if numHeadsKV < numHeadsQ {
		if numHeadsQ%numHeadsKV != 0 {
			return nil, nil, errors.Errorf("FusedScaledDotProductAttention: query heads (%d) not divisible by key heads (%d)", numHeadsQ, numHeadsKV)
		}
		repeats := numHeadsQ / numHeadsKV
		sliceK := make([]compute.Value, repeats)
		sliceV := make([]compute.Value, repeats)
		for i := 0; i < repeats; i++ {
			sliceK[i] = kBHSD
			sliceV[i] = vBHSD
		}
		kBHSDVal, err := f.Concatenate(1, sliceK...)
		if err != nil {
			return nil, nil, err
		}
		vBHSDVal, err := f.Concatenate(1, sliceV...)
		if err != nil {
			return nil, nil, err
		}
		kBHSD = kBHSDVal.(*Node)
		vBHSD = vBHSDVal.(*Node)
	}

	// Q @ K^T -> [B, H, S, Skv]
	scoresVal, err := f.DotGeneral(qBHSD, []int{3}, []int{0, 1}, kBHSD, []int{3}, []int{0, 1}, compute.DotGeneralConfig{})
	if err != nil {
		return nil, nil, errors.Wrap(err, "FusedScaledDotProductAttention: q @ k^T failed")
	}

	headDim := qBHSD.shape.Dimensions[3]
	scale := 1.0 / math.Sqrt(float64(headDim))
	if options != nil && options.Scale != 0 {
		scale = options.Scale
	}

	var scaleConst compute.Value
	switch qNode.shape.DType {
	case dtypes.Float32:
		scaleConst, err = f.Constant([]float32{float32(scale)})
	case dtypes.Float64:
		scaleConst, err = f.Constant([]float64{scale})
	case dtypes.Float16:
		scaleConst, err = f.Constant([]float16.Float16{float16.FromFloat32(float32(scale))})
	case dtypes.BFloat16:
		scaleConst, err = f.Constant([]bfloat16.BFloat16{bfloat16.FromFloat32(float32(scale))})
	default:
		return nil, nil, errors.Errorf("FusedScaledDotProductAttention: unsupported dtype %s", qNode.shape.DType)
	}
	if err != nil {
		return nil, nil, err
	}

	scoresVal, err = f.Mul(scoresVal, scaleConst)
	if err != nil {
		return nil, nil, err
	}

	// Apply causal mask if requested
	if options != nil && options.Causal {
		seqLen := qBHSD.shape.Dimensions[2]
		kvLen := kBHSD.shape.Dimensions[2]

		var causalMaskConst compute.Value
		switch qNode.shape.DType {
		case dtypes.Float32:
			maskFlat := make([]float32, seqLen*kvLen)
			for i := 0; i < seqLen; i++ {
				for j := 0; j < kvLen; j++ {
					if j > i {
						maskFlat[i*kvLen+j] = -1e9
					}
				}
			}
			causalMaskConst, err = f.Constant(maskFlat, 1, 1, seqLen, kvLen)
		case dtypes.Float64:
			maskFlat := make([]float64, seqLen*kvLen)
			for i := 0; i < seqLen; i++ {
				for j := 0; j < kvLen; j++ {
					if j > i {
						maskFlat[i*kvLen+j] = -1e9
					}
				}
			}
			causalMaskConst, err = f.Constant(maskFlat, 1, 1, seqLen, kvLen)
		case dtypes.Float16:
			maskFlat := make([]float16.Float16, seqLen*kvLen)
			negVal := float16.FromFloat32(-10000.0)
			for i := 0; i < seqLen; i++ {
				for j := 0; j < kvLen; j++ {
					if j > i {
						maskFlat[i*kvLen+j] = negVal
					}
				}
			}
			causalMaskConst, err = f.Constant(maskFlat, 1, 1, seqLen, kvLen)
		case dtypes.BFloat16:
			maskFlat := make([]bfloat16.BFloat16, seqLen*kvLen)
			negVal := bfloat16.FromFloat32(-1e9)
			for i := 0; i < seqLen; i++ {
				for j := 0; j < kvLen; j++ {
					if j > i {
						maskFlat[i*kvLen+j] = negVal
					}
				}
			}
			causalMaskConst, err = f.Constant(maskFlat, 1, 1, seqLen, kvLen)
		}
		if err != nil {
			return nil, nil, err
		}
		scoresVal, err = f.Add(scoresVal, causalMaskConst)
		if err != nil {
			return nil, nil, err
		}
	}

	// Apply explicit mask if provided
	if options != nil && options.Mask != nil {
		mNode, ok := options.Mask.(*Node)
		if !ok {
			return nil, nil, errors.New("FusedScaledDotProductAttention: Mask must be a valid onnxruntime node")
		}
		scoresNode := scoresVal.(*Node)
		maskReshaped, err := expandRankToMatch(f, mNode, scoresNode.shape.Rank())
		if err != nil {
			return nil, nil, errors.Wrap(err, "FusedScaledDotProductAttention: mask reshape failed")
		}
		maskNode := maskReshaped.(*Node)

		if maskNode.shape.DType == dtypes.Bool {
			var negInfConst compute.Value
			switch qNode.shape.DType {
			case dtypes.Float32:
				negInfConst, err = f.Constant([]float32{-1e9})
			case dtypes.Float64:
				negInfConst, err = f.Constant([]float64{-1e9})
			case dtypes.Float16:
				negInfConst, err = f.Constant([]float16.Float16{float16.FromFloat32(-10000.0)})
			case dtypes.BFloat16:
				negInfConst, err = f.Constant([]bfloat16.BFloat16{bfloat16.FromFloat32(-1e9)})
			}
			if err != nil {
				return nil, nil, err
			}
			scoresVal, err = f.Where(maskNode, scoresVal, negInfConst)
			if err != nil {
				return nil, nil, err
			}
		} else {
			scoresVal, err = f.Add(scoresVal, maskNode)
			if err != nil {
				return nil, nil, err
			}
		}
	}

	// Apply additive bias if provided
	if options != nil && options.Bias != nil {
		biasNode, ok := options.Bias.(*Node)
		if !ok {
			return nil, nil, errors.New("FusedScaledDotProductAttention: Bias must be a valid onnxruntime node")
		}
		scoresNode := scoresVal.(*Node)
		biasReshaped, err := expandRankToMatch(f, biasNode, scoresNode.shape.Rank())
		if err != nil {
			return nil, nil, errors.Wrap(err, "FusedScaledDotProductAttention: bias reshape failed")
		}
		scoresVal, err = f.Add(scoresVal, biasReshaped)
		if err != nil {
			return nil, nil, err
		}
	}

	// Softmax along axis 3 (Skv)
	attnWeights, err := f.FusedSoftmax(scoresVal, 3)
	if err != nil {
		return nil, nil, errors.Wrap(err, "FusedScaledDotProductAttention: Softmax failed")
	}

	// Attn @ V -> [B, H, S, D]
	outBHSD, err := f.DotGeneral(attnWeights, []int{3}, []int{0, 1}, vBHSD, []int{2}, []int{0, 1}, compute.DotGeneralConfig{})
	if err != nil {
		return nil, nil, errors.Wrap(err, "FusedScaledDotProductAttention: attn @ v failed")
	}

	var finalOut compute.Value = outBHSD
	if axesLayout == compute.AttentionAxesLayoutBSHD {
		finalOut, err = f.Transpose(outBHSD, 0, 2, 1, 3)
		if err != nil {
			return nil, nil, err
		}
	}

	return finalOut, nil, nil
}

// FusedScaledDotProductAttentionVJP computes gradients of FusedScaledDotProductAttention.
func (f *Function) FusedScaledDotProductAttentionVJP(
	query, key, value compute.Value,
	axesLayout compute.AttentionAxesLayout,
	options *compute.ScaledDotProductAttentionConfig,
	output compute.Value, statesForVJP []compute.Value, dOutput compute.Value) (dQuery, dKey, dValue compute.Value, err error) {
	return nil, nil, nil, errors.Wrap(compute.ErrNotImplemented, "FusedScaledDotProductAttentionVJP is not implemented for onnxruntime backend")
}

// QuantizedEmbeddingLookup performs a quantized embedding lookup with on-the-fly dequantization.
func (f *Function) QuantizedEmbeddingLookup(data, indices compute.Value, dataQuantization *compute.Quantization) (compute.Value, error) {
	return nil, errors.Wrap(compute.ErrNotImplemented, "QuantizedEmbeddingLookup is not implemented for onnxruntime backend")
}

// FusedQuantizedDense performs fused dequantization + matmul + optional bias + optional activation.
func (f *Function) FusedQuantizedDense(x, weights, bias compute.Value, weightsQuantization *compute.Quantization, activation compute.ActivationType) (compute.Value, error) {
	if weightsQuantization == nil {
		return nil, errors.New("FusedQuantizedDense: weightsQuantization must not be nil")
	}

	if weightsQuantization.Scheme != compute.QuantLinear {
		return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedQuantizedDense: scheme %v not supported for onnxruntime backend", weightsQuantization.Scheme)
	}

	xNode, ok1 := x.(*Node)
	wNode, ok2 := weights.(*Node)
	if !ok1 || !ok2 {
		return nil, errors.New("FusedQuantizedDense: inputs must be valid onnxruntime nodes")
	}

	// Dequantize weights: floatWeight = (weights - zeroPoint) * scale
	wFloat, err := f.ConvertDType(wNode, dtypes.Float32)
	if err != nil {
		return nil, errors.Wrap(err, "FusedQuantizedDense: converting weights to float32 failed")
	}

	if weightsQuantization.ZeroPoint != nil {
		zpFloat, err := f.ConvertDType(weightsQuantization.ZeroPoint, dtypes.Float32)
		if err != nil {
			return nil, errors.Wrap(err, "FusedQuantizedDense: converting zeroPoint to float32 failed")
		}
		wNode := wFloat.(*Node)
		zpReshaped, err := expandRankToMatch(f, zpFloat, wNode.shape.Rank())
		if err != nil {
			return nil, errors.Wrap(err, "FusedQuantizedDense: zeroPoint reshape failed")
		}
		wFloatVal, err := f.Sub(wFloat, zpReshaped)
		if err != nil {
			return nil, errors.Wrap(err, "FusedQuantizedDense: sub zeroPoint failed")
		}
		wFloat = wFloatVal
	}

	if weightsQuantization.Scale != nil {
		scaleFloat, err := f.ConvertDType(weightsQuantization.Scale, dtypes.Float32)
		if err != nil {
			return nil, errors.Wrap(err, "FusedQuantizedDense: converting scale to float32 failed")
		}
		wNode := wFloat.(*Node)
		scaleReshaped, err := expandRankToMatch(f, scaleFloat, wNode.shape.Rank())
		if err != nil {
			return nil, errors.Wrap(err, "FusedQuantizedDense: scale reshape failed")
		}
		wFloatVal, err := f.Mul(wFloat, scaleReshaped)
		if err != nil {
			return nil, errors.Wrap(err, "FusedQuantizedDense: mul scale failed")
		}
		wFloat = wFloatVal
	}

	return f.FusedDense(xNode, wFloat, bias, compute.DenseConfig{
		Activation:   activation,
		WeightLayout: compute.DenseLayoutInputOutputs,
	})
}

// FusedAttentionQKVProjection performs fused Query-Key-Value projection.
func (f *Function) FusedAttentionQKVProjection(
	x, wQKV, biasQ, biasK, biasV compute.Value,
	queryDim, keyValueDim int) (query, key, value compute.Value, err error) {
	xNode, ok1 := x.(*Node)
	wNode, ok2 := wQKV.(*Node)
	if !ok1 || !ok2 {
		return nil, nil, nil, errors.New("FusedAttentionQKVProjection: inputs must be valid onnxruntime nodes")
	}

	xRank := xNode.shape.Rank()
	if xRank != 2 {
		return nil, nil, nil, errors.Errorf("FusedAttentionQKVProjection: x must be 2D [batch, inFeatures], got rank %d", xRank)
	}

	// Compute matmul: y = x @ wQKV -> shape [batch, queryDim + 2*keyValueDim]
	yVal, err := f.DotGeneral(xNode, []int{1}, nil, wNode, []int{0}, nil, compute.DotGeneralConfig{})
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: matmul failed")
	}

	yNode, ok := yVal.(*Node)
	if !ok {
		return nil, nil, nil, errors.New("FusedAttentionQKVProjection: matmul output is not a valid onnxruntime node")
	}

	batchDim := yNode.shape.Dimensions[0]
	totalDim := queryDim + 2*keyValueDim

	// Slice Q, K, V along axis 1
	qSlice, err := f.Slice(yVal, []int{0, 0}, []int{batchDim, queryDim}, []int{1, 1})
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: slicing Q failed")
	}

	kSlice, err := f.Slice(yVal, []int{0, queryDim}, []int{batchDim, queryDim + keyValueDim}, []int{1, 1})
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: slicing K failed")
	}

	vSlice, err := f.Slice(yVal, []int{0, queryDim + keyValueDim}, []int{batchDim, totalDim}, []int{1, 1})
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: slicing V failed")
	}

	var queryRes, keyRes, valueRes compute.Value = qSlice, kSlice, vSlice

	if biasQ != nil {
		bqNode, ok := biasQ.(*Node)
		if !ok {
			return nil, nil, nil, errors.New("FusedAttentionQKVProjection: biasQ must be a valid onnxruntime node")
		}
		qNode := qSlice.(*Node)
		bqReshaped, err := expandRankToMatch(f, bqNode, qNode.shape.Rank())
		if err != nil {
			return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: biasQ reshape failed")
		}
		queryRes, err = f.Add(queryRes, bqReshaped)
		if err != nil {
			return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: adding biasQ failed")
		}
	}

	if biasK != nil {
		bkNode, ok := biasK.(*Node)
		if !ok {
			return nil, nil, nil, errors.New("FusedAttentionQKVProjection: biasK must be a valid onnxruntime node")
		}
		kNode := kSlice.(*Node)
		bkReshaped, err := expandRankToMatch(f, bkNode, kNode.shape.Rank())
		if err != nil {
			return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: biasK reshape failed")
		}
		keyRes, err = f.Add(keyRes, bkReshaped)
		if err != nil {
			return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: adding biasK failed")
		}
	}

	if biasV != nil {
		bvNode, ok := biasV.(*Node)
		if !ok {
			return nil, nil, nil, errors.New("FusedAttentionQKVProjection: biasV must be a valid onnxruntime node")
		}
		vNode := vSlice.(*Node)
		bvReshaped, err := expandRankToMatch(f, bvNode, vNode.shape.Rank())
		if err != nil {
			return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: biasV reshape failed")
		}
		valueRes, err = f.Add(valueRes, bvReshaped)
		if err != nil {
			return nil, nil, nil, errors.Wrap(err, "FusedAttentionQKVProjection: adding biasV failed")
		}
	}

	return queryRes, keyRes, valueRes, nil
}
