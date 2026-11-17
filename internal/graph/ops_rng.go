// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute-onnx/internal/executionprovider"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// RNGBitGenerator generates a tensor of the requested shape filled with uniform random bits
// using the specified algorithm ([compute.RNGDefault], [compute.RNGSplitMix64], or [compute.RNGThreeFry]).
func (f *Function) RNGBitGenerator(
	state compute.Value,
	shape shapes.Shape,
	algorithm compute.RNGAlgorithm,
) (newState compute.Value, values compute.Value, err error) {
	stateNode, ok := state.(*Node)
	if !ok {
		return nil, nil, errors.New("RNGBitGenerator: state must be a valid onnxruntime node")
	}

	var caps compute.Capabilities
	if f.builder != nil && f.builder.backend != nil {
		caps = f.builder.backend.Capabilities()
	} else {
		caps = compute.Capabilities{
			Operations: map[compute.OpType]bool{compute.OpTypeRNGBitGenerator: true},
			RNGAlgorithms: map[compute.RNGAlgorithm]bool{
				compute.RNGDefault:    true,
				compute.RNGSplitMix64: true,
				compute.RNGThreeFry:   true,
			},
			DefaultRNGAlgorithm: compute.RNGSplitMix64,
		}
		if f.executionProvider == executionprovider.WebGPU {
			caps.DefaultRNGAlgorithm = compute.RNGDefault
			caps.DefaultRNGStateShape = shapes.Make(dtypes.Uint32, 4)
		}
	}

	expectedStateShape, err := caps.RNGStateShape(algorithm)
	if err != nil {
		return nil, nil, err
	}
	if _, _, err := shapeinference.RNGBitGenerator(expectedStateShape, stateNode.shape, shape); err != nil {
		return nil, nil, err
	}

	resolved := caps.ResolveRNGAlgorithm(algorithm)
	switch resolved {
	case compute.RNGSplitMix64:
		return f.rngSplitMix64(stateNode, shape)
	case compute.RNGThreeFry:
		return f.rngThreeFry(stateNode, shape, false)
	case compute.RNGDefault:
		if stateNode.shape.DType == dtypes.Uint32 {
			return f.rngThreeFry(stateNode, shape, true)
		}
		return nil, nil, errors.Wrapf(compute.ErrNotImplemented, "RNGBitGenerator: unsupported RNGDefault state shape %s", stateNode.shape)
	default:
		return nil, nil, errors.Wrapf(compute.ErrNotImplemented, "RNGBitGenerator: RNG algorithm %s is not supported on ONNX Runtime backend", algorithm)
	}
}

func (f *Function) rngSliceScalar(arr *Node, idx int) (*Node, error) {
	val, err := f.Slice(arr, []int{idx}, []int{idx + 1}, []int{1})
	if err != nil {
		return nil, err
	}
	reshaped, err := f.Reshape(val)
	if err != nil {
		return nil, err
	}
	return reshaped.(*Node), nil
}

func (f *Function) rngTotalSizeAndSpecs(shape shapes.Shape) (
	totalSizeInt64 compute.Value,
	dynSpecs []compute.DynamicDimensionSpec,
	emptyVal compute.Value,
	isStaticEmpty bool,
	err error,
) {
	if !shape.IsDynamic() {
		size := shape.Size()
		if size == 0 {
			emptyVal, err = f.Constant(dtypes.MakeAnySlice(shape.DType, 0), shape.Dimensions...)
			return nil, nil, emptyVal, true, err
		}
		totalSizeInt64, err = f.Constant([]int64{int64(size)})
		if err != nil {
			return nil, nil, nil, false, err
		}
		// Make it scalar Int64 to match dynamic path.
		totalSizeInt64, err = f.Reshape(totalSizeInt64)
		return totalSizeInt64, nil, nil, false, err
	}

	dynSpecs = make([]compute.DynamicDimensionSpec, shape.Rank())
	staticProd := int64(1)
	var dynSizeInt64 compute.Value
	for i, dim := range shape.Dimensions {
		if dim != shapes.DynamicDim {
			dynSpecs[i] = compute.DynamicDimensionSpec{Static: dim}
			staticProd *= int64(dim)
			continue
		}
		axisName := shape.AxisName(i)
		dim1D, ok := f.resolveDynamicAxisDim1D(axisName)
		if !ok {
			return nil, nil, nil, false, errors.Errorf("RNGBitGenerator: unknown dynamic axis %q at axis %d of shape %s", axisName, i, shape)
		}
		dimScalar, err := f.Reshape(dim1D)
		if err != nil {
			return nil, nil, nil, false, err
		}
		dynSpecs[i] = compute.DynamicDimensionSpec{Name: axisName, Value: dimScalar}
		if dynSizeInt64 == nil {
			dynSizeInt64 = dimScalar
		} else {
			dynSizeInt64, err = f.Mul(dynSizeInt64, dimScalar)
			if err != nil {
				return nil, nil, nil, false, err
			}
		}
	}
	staticConst, err := f.Constant([]int64{staticProd})
	if err != nil {
		return nil, nil, nil, false, err
	}
	totalSizeInt64 = staticConst
	if dynSizeInt64 != nil {
		totalSizeInt64, err = f.Mul(dynSizeInt64, staticConst)
		if err != nil {
			return nil, nil, nil, false, err
		}
	}
	return totalSizeInt64, dynSpecs, nil, false, nil
}

func (f *Function) rngReshapeFlatOutput(flat compute.Value, shape shapes.Shape, dynSpecs []compute.DynamicDimensionSpec) (compute.Value, error) {
	if shape.IsDynamic() {
		return f.DynamicReshape(flat, dynSpecs...)
	}
	if shape.Rank() == 0 {
		return f.Reshape(flat)
	}
	return f.Reshape(flat, shape.Dimensions...)
}

// rngSplitMix64 implements canonical SplitMix64 on [2]uint64 ([key, counter]).
func (f *Function) rngSplitMix64(stateNode *Node, shape shapes.Shape) (newState, values compute.Value, err error) {
	key, err := f.rngSliceScalar(stateNode, 0)
	if err != nil {
		return nil, nil, err
	}
	counter, err := f.rngSliceScalar(stateNode, 1)
	if err != nil {
		return nil, nil, err
	}

	totalSizeInt64, dynSpecs, emptyVal, isStaticEmpty, err := f.rngTotalSizeAndSpecs(shape)
	if err != nil {
		return nil, nil, err
	}
	if isStaticEmpty {
		return stateNode, emptyVal, nil
	}

	var iotaVal compute.Value
	if shape.IsDynamic() {
		iotaVal, err = f.DynamicIota(dtypes.Uint64, 0, compute.DynamicDimensionSpec{
			Name:  shapes.UniqueAxisName("#rng_flat"),
			Value: totalSizeInt64,
		})
	} else {
		iotaVal, err = f.Iota(shapes.Make(dtypes.Uint64, shape.Size()), 0)
	}
	if err != nil {
		return nil, nil, err
	}
	sizeUint64, err := f.ConvertDType(totalSizeInt64, dtypes.Uint64)
	if err != nil {
		return nil, nil, err
	}

	oneU64, err := f.Constant([]uint64{1})
	if err != nil {
		return nil, nil, err
	}
	gammaConst, err := f.Constant([]uint64{0x9e3779b97f4a7c15})
	if err != nil {
		return nil, nil, err
	}
	c1Const, err := f.Constant([]uint64{0xbf58476d1ce4e5b9})
	if err != nil {
		return nil, nil, err
	}
	c2Const, err := f.Constant([]uint64{0x94d049bb133111eb})
	if err != nil {
		return nil, nil, err
	}

	idxPlus1, err := f.Add(iotaVal, oneU64)
	if err != nil {
		return nil, nil, err
	}
	ctrIdx, err := f.Add(idxPlus1, counter)
	if err != nil {
		return nil, nil, err
	}
	offset, err := f.Mul(ctrIdx, gammaConst)
	if err != nil {
		return nil, nil, err
	}
	z, err := f.Add(key, offset)
	if err != nil {
		return nil, nil, err
	}

	mixStep := func(val compute.Value, shift int, c compute.Value) (compute.Value, error) {
		shiftConst, err := f.Constant([]uint64{uint64(shift)})
		if err != nil {
			return nil, err
		}
		shifted, err := f.ShiftRightLogical(val, shiftConst)
		if err != nil {
			return nil, err
		}
		xored, err := f.BitwiseXor(val, shifted)
		if err != nil {
			return nil, err
		}
		return f.Mul(xored, c)
	}

	z, err = mixStep(z, 30, c1Const)
	if err != nil {
		return nil, nil, err
	}
	z, err = mixStep(z, 27, c2Const)
	if err != nil {
		return nil, nil, err
	}
	shift31Const, err := f.Constant([]uint64{31})
	if err != nil {
		return nil, nil, err
	}
	shifted31, err := f.ShiftRightLogical(z, shift31Const)
	if err != nil {
		return nil, nil, err
	}
	flatValues, err := f.BitwiseXor(z, shifted31)
	if err != nil {
		return nil, nil, err
	}

	castedValues := flatValues
	if shape.DType != dtypes.Uint64 {
		castedValues, err = f.ConvertDType(flatValues, shape.DType)
		if err != nil {
			return nil, nil, err
		}
	}
	values, err = f.rngReshapeFlatOutput(castedValues, shape, dynSpecs)
	if err != nil {
		return nil, nil, err
	}

	newCounter, err := f.Add(counter, sizeUint64)
	if err != nil {
		return nil, nil, err
	}
	key1D, err := f.Reshape(key, 1)
	if err != nil {
		return nil, nil, err
	}
	newCtr1D, err := f.Reshape(newCounter, 1)
	if err != nil {
		return nil, nil, err
	}
	newState, err = f.Concatenate(0, key1D, newCtr1D)
	if err != nil {
		return nil, nil, err
	}
	return newState, values, nil
}

// rngThreeFry implements OpenXLA-compatible ThreeFry-2x32-20 on either [2]uint64 (stateU32=false)
// or pure 32-bit [4]uint32 (stateU32=true, used for WebGPU RNGDefault).
func (f *Function) rngThreeFry(stateNode *Node, shape shapes.Shape, stateU32 bool) (newState, values compute.Value, err error) {
	var key0, key1, ctr0, ctr1 *Node
	if stateU32 {
		if key0, err = f.rngSliceScalar(stateNode, 0); err != nil {
			return nil, nil, err
		}
		if key1, err = f.rngSliceScalar(stateNode, 1); err != nil {
			return nil, nil, err
		}
		if ctr0, err = f.rngSliceScalar(stateNode, 2); err != nil {
			return nil, nil, err
		}
		if ctr1, err = f.rngSliceScalar(stateNode, 3); err != nil {
			return nil, nil, err
		}
	} else {
		key64, err := f.rngSliceScalar(stateNode, 0)
		if err != nil {
			return nil, nil, err
		}
		ctr64, err := f.rngSliceScalar(stateNode, 1)
		if err != nil {
			return nil, nil, err
		}
		shift32, err := f.Constant([]uint64{32})
		if err != nil {
			return nil, nil, err
		}
		split64 := func(v *Node) (*Node, *Node, error) {
			lo, err := f.ConvertDType(v, dtypes.Uint32)
			if err != nil {
				return nil, nil, err
			}
			hi64, err := f.ShiftRightLogical(v, shift32)
			if err != nil {
				return nil, nil, err
			}
			hi, err := f.ConvertDType(hi64, dtypes.Uint32)
			if err != nil {
				return nil, nil, err
			}
			return lo.(*Node), hi.(*Node), nil
		}
		if key0, key1, err = split64(key64); err != nil {
			return nil, nil, err
		}
		if ctr0, ctr1, err = split64(ctr64); err != nil {
			return nil, nil, err
		}
	}

	totalSizeInt64, dynSpecs, emptyVal, isStaticEmpty, err := f.rngTotalSizeAndSpecs(shape)
	if err != nil {
		return nil, nil, err
	}
	if isStaticEmpty {
		return stateNode, emptyVal, nil
	}

	var wordsPerElem int64 = 1
	if shape.DType == dtypes.Uint64 {
		wordsPerElem = 2
	}
	var numWordsInt64 compute.Value = totalSizeInt64
	if wordsPerElem == 2 {
		twoI64, err := f.Constant([]int64{2})
		if err != nil {
			return nil, nil, err
		}
		numWordsInt64, err = f.Mul(totalSizeInt64, twoI64)
		if err != nil {
			return nil, nil, err
		}
	}
	oneI64, err := f.Constant([]int64{1})
	if err != nil {
		return nil, nil, err
	}
	twoI64, err := f.Constant([]int64{2})
	if err != nil {
		return nil, nil, err
	}
	numWordsPlus1, err := f.Add(numWordsInt64, oneI64)
	if err != nil {
		return nil, nil, err
	}
	numBlocksInt64, err := f.Div(numWordsPlus1, twoI64)
	if err != nil {
		return nil, nil, err
	}

	var blockIota compute.Value
	if shape.IsDynamic() {
		blockIota, err = f.DynamicIota(dtypes.Uint32, 0, compute.DynamicDimensionSpec{
			Name:  shapes.UniqueAxisName("#rng_blocks"),
			Value: numBlocksInt64,
		})
	} else {
		numBlocks := (shape.Size()*int(wordsPerElem) + 1) / 2
		blockIota, err = f.Iota(shapes.Make(dtypes.Uint32, numBlocks), 0)
	}
	if err != nil {
		return nil, nil, err
	}

	// 64-bit block counter = (ctr0, ctr1) + blockIota (mod 2^64).
	c0, err := f.Add(blockIota, ctr0)
	if err != nil {
		return nil, nil, err
	}
	carryBool, err := f.LessThan(c0, ctr0)
	if err != nil {
		return nil, nil, err
	}
	carryU32, err := f.ConvertDType(carryBool, dtypes.Uint32)
	if err != nil {
		return nil, nil, err
	}
	c1, err := f.Add(ctr1, carryU32)
	if err != nil {
		return nil, nil, err
	}

	// Key schedule.
	magicConst, err := f.Constant([]uint32{0x1BD11BDA})
	if err != nil {
		return nil, nil, err
	}
	k0XorK1, err := f.BitwiseXor(key0, key1)
	if err != nil {
		return nil, nil, err
	}
	key2Val, err := f.BitwiseXor(k0XorK1, magicConst)
	if err != nil {
		return nil, nil, err
	}
	ks := [3]compute.Value{key0, key1, key2Val}

	x0, err := f.Add(c0, ks[0])
	if err != nil {
		return nil, nil, err
	}
	x1, err := f.Add(c1, ks[1])
	if err != nil {
		return nil, nil, err
	}

	rotations := [8]int{13, 15, 26, 6, 17, 29, 16, 24}
	var rotLeftConsts, rotRightConsts [8]compute.Value
	for i, r := range rotations {
		if rotLeftConsts[i], err = f.Constant([]uint32{uint32(r)}); err != nil {
			return nil, nil, err
		}
		if rotRightConsts[i], err = f.Constant([]uint32{uint32(32 - r)}); err != nil {
			return nil, nil, err
		}
	}

	for r := range 20 {
		x0, err = f.Add(x0, x1)
		if err != nil {
			return nil, nil, err
		}
		ri := r % 8
		shl, err := f.ShiftLeft(x1, rotLeftConsts[ri])
		if err != nil {
			return nil, nil, err
		}
		shr, err := f.ShiftRightLogical(x1, rotRightConsts[ri])
		if err != nil {
			return nil, nil, err
		}
		rot, err := f.BitwiseOr(shl, shr)
		if err != nil {
			return nil, nil, err
		}
		x1, err = f.BitwiseXor(rot, x0)
		if err != nil {
			return nil, nil, err
		}
		if (r+1)%4 == 0 {
			inj := (r + 1) / 4
			if x0, err = f.Add(x0, ks[inj%3]); err != nil {
				return nil, nil, err
			}
			injConst, err := f.Constant([]uint32{uint32(inj)})
			if err != nil {
				return nil, nil, err
			}
			kInj, err := f.Add(ks[(inj+1)%3], injConst)
			if err != nil {
				return nil, nil, err
			}
			if x1, err = f.Add(x1, kInj); err != nil {
				return nil, nil, err
			}
		}
	}

	// Interleave x0 and x1 -> [numBlocks, 2] -> [2*numBlocks] -> slice [:numWords].
	x0Col, err := f.rngExpandTrailing1(x0)
	if err != nil {
		return nil, nil, err
	}
	x1Col, err := f.rngExpandTrailing1(x1)
	if err != nil {
		return nil, nil, err
	}
	paired, err := f.Concatenate(1, x0Col, x1Col)
	if err != nil {
		return nil, nil, err
	}

	var flatWords compute.Value
	if shape.IsDynamic() {
		totalWordsSpec := compute.DynamicDimensionSpec{Name: shapes.UniqueAxisName("#rng_2blocks")}
		flat2Blocks, err := f.DynamicReshape(paired, totalWordsSpec)
		if err != nil {
			return nil, nil, err
		}
		flatWords, err = f.rngSlice1D(flat2Blocks.(*Node), numWordsInt64.(*Node))
		if err != nil {
			return nil, nil, err
		}
	} else {
		numBlocks := (shape.Size()*int(wordsPerElem) + 1) / 2
		numWords := shape.Size() * int(wordsPerElem)
		flat2Blocks, err := f.Reshape(paired, numBlocks*2)
		if err != nil {
			return nil, nil, err
		}
		if numWords < numBlocks*2 {
			flatWords, err = f.Slice(flat2Blocks, []int{0}, []int{numWords}, []int{1})
			if err != nil {
				return nil, nil, err
			}
		} else {
			flatWords = flat2Blocks
		}
	}

	var finalFlat compute.Value
	if shape.DType == dtypes.Uint64 {
		// Reshape flatWords [2*N] -> [N, 2], combine lo | (hi << 32) into [N]Uint64.
		var pairs compute.Value
		if shape.IsDynamic() {
			pairs, err = f.DynamicReshape(flatWords,
				compute.DynamicDimensionSpec{Name: shapes.UniqueAxisName("#rng_n"), Value: totalSizeInt64},
				compute.DynamicDimensionSpec{Static: 2},
			)
		} else {
			pairs, err = f.Reshape(flatWords, shape.Size(), 2)
		}
		if err != nil {
			return nil, nil, err
		}
		lo32, err := f.rngSliceAxis1(pairs.(*Node), 0, 1)
		if err != nil {
			return nil, nil, err
		}
		hi32, err := f.rngSliceAxis1(pairs.(*Node), 1, 2)
		if err != nil {
			return nil, nil, err
		}
		lo64, err := f.ConvertDType(lo32, dtypes.Uint64)
		if err != nil {
			return nil, nil, err
		}
		hi64, err := f.ConvertDType(hi32, dtypes.Uint64)
		if err != nil {
			return nil, nil, err
		}
		shift32U64, err := f.Constant([]uint64{32})
		if err != nil {
			return nil, nil, err
		}
		hiShifted, err := f.ShiftLeft(hi64, shift32U64)
		if err != nil {
			return nil, nil, err
		}
		combined2D, err := f.BitwiseOr(lo64, hiShifted)
		if err != nil {
			return nil, nil, err
		}
		if shape.IsDynamic() {
			finalFlat, err = f.DynamicReshape(combined2D, compute.DynamicDimensionSpec{Name: shapes.UniqueAxisName("#rng_u64"), Value: totalSizeInt64})
		} else {
			finalFlat, err = f.Reshape(combined2D, shape.Size())
		}
		if err != nil {
			return nil, nil, err
		}
	} else {
		finalFlat = flatWords
		if shape.DType != dtypes.Uint32 {
			finalFlat, err = f.ConvertDType(flatWords, shape.DType)
			if err != nil {
				return nil, nil, err
			}
		}
	}

	values, err = f.rngReshapeFlatOutput(finalFlat, shape, dynSpecs)
	if err != nil {
		return nil, nil, err
	}

	// Update state counter by numBlocks.
	if stateU32 {
		numBlocksU32, err := f.ConvertDType(numBlocksInt64, dtypes.Uint32)
		if err != nil {
			return nil, nil, err
		}
		newCtr0, err := f.Add(ctr0, numBlocksU32)
		if err != nil {
			return nil, nil, err
		}
		ctrCarryBool, err := f.LessThan(newCtr0, ctr0)
		if err != nil {
			return nil, nil, err
		}
		ctrCarryU32, err := f.ConvertDType(ctrCarryBool, dtypes.Uint32)
		if err != nil {
			return nil, nil, err
		}
		newCtr1, err := f.Add(ctr1, ctrCarryU32)
		if err != nil {
			return nil, nil, err
		}
		k01D, _ := f.Reshape(key0, 1)
		k11D, _ := f.Reshape(key1, 1)
		c01D, _ := f.Reshape(newCtr0, 1)
		c11D, _ := f.Reshape(newCtr1, 1)
		newState, err = f.Concatenate(0, k01D, k11D, c01D, c11D)
		if err != nil {
			return nil, nil, err
		}
		return newState, values, nil
	}

	key64, err := f.rngSliceScalar(stateNode, 0)
	if err != nil {
		return nil, nil, err
	}
	ctr64, err := f.rngSliceScalar(stateNode, 1)
	if err != nil {
		return nil, nil, err
	}
	numBlocksU64, err := f.ConvertDType(numBlocksInt64, dtypes.Uint64)
	if err != nil {
		return nil, nil, err
	}
	newCtr64, err := f.Add(ctr64, numBlocksU64)
	if err != nil {
		return nil, nil, err
	}
	key1D, err := f.Reshape(key64, 1)
	if err != nil {
		return nil, nil, err
	}
	ctr1D, err := f.Reshape(newCtr64, 1)
	if err != nil {
		return nil, nil, err
	}
	newState, err = f.Concatenate(0, key1D, ctr1D)
	if err != nil {
		return nil, nil, err
	}
	return newState, values, nil
}

func (f *Function) rngExpandTrailing1(v compute.Value) (compute.Value, error) {
	n := v.(*Node)
	if n.shape.IsDynamic() {
		return f.DynamicReshape(n,
			compute.DynamicDimensionSpec{Name: n.shape.AxisName(0)},
			compute.DynamicDimensionSpec{Static: 1},
		)
	}
	return f.Reshape(n, n.shape.Dimensions[0], 1)
}

func (f *Function) rngSliceAxis1(x *Node, start, end int) (compute.Value, error) {
	if !x.shape.IsDynamic() {
		return f.Slice(x, []int{0, start}, []int{x.shape.Dimensions[0], end}, []int{1, 1})
	}
	startsConst, err := f.Constant([]int64{int64(start)}, 1)
	if err != nil {
		return nil, err
	}
	endsConst, err := f.Constant([]int64{int64(end)}, 1)
	if err != nil {
		return nil, err
	}
	axesConst, err := f.Constant([]int64{1}, 1)
	if err != nil {
		return nil, err
	}
	stepsConst, err := f.Constant([]int64{1}, 1)
	if err != nil {
		return nil, err
	}
	outShape := x.shape.Clone()
	outShape.Dimensions[1] = end - start
	return f.addNode(&Node{
		opType: "Slice",
		inputs: []*Node{x, startsConst.(*Node), endsConst.(*Node), axesConst.(*Node), stepsConst.(*Node)},
		shape:  outShape,
	}), nil
}

func (f *Function) rngSlice1D(x, endScalar *Node) (compute.Value, error) {
	startsConst, err := f.Constant([]int64{0}, 1)
	if err != nil {
		return nil, err
	}
	ends1D, err := f.Reshape(endScalar, 1)
	if err != nil {
		return nil, err
	}
	axesConst, err := f.Constant([]int64{0}, 1)
	if err != nil {
		return nil, err
	}
	stepsConst, err := f.Constant([]int64{1}, 1)
	if err != nil {
		return nil, err
	}
	outShape := shapes.MakeDynamic(x.shape.DType, []int{shapes.DynamicDim}, []string{shapes.UniqueAxisName("#rng_words")})
	node := &Node{
		opType: "Slice",
		inputs: []*Node{x, startsConst.(*Node), ends1D.(*Node), axesConst.(*Node), stepsConst.(*Node)},
		shape:  outShape,
	}
	return f.addNode(node), nil
}
