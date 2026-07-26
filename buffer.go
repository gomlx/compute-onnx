// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxruntime

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
	ort "github.com/gomlx/compute-onnx/internal/ort"
)

type ortTensorWrapper interface {
	GetShape() ort.Shape
	Destroy() error
	Value() ort.Value
	ToFlatData(flat any) error
	CopyFrom(flat any) error
}

type typedTensor[T ort.TensorData] struct {
	tensor *ort.Tensor[T]
}

func (t *typedTensor[T]) GetShape() ort.Shape {
	return t.tensor.GetShape()
}

func (t *typedTensor[T]) Destroy() error {
	return t.tensor.Destroy()
}

func (t *typedTensor[T]) Value() ort.Value {
	return t.tensor
}

func (t *typedTensor[T]) ToFlatData(flat any) error {
	slice, ok := flat.([]T)
	if !ok {
		return errors.Errorf("flat data type %T does not match tensor type %T", flat, slice)
	}
	copy(slice, t.tensor.GetData())
	return nil
}

func (t *typedTensor[T]) CopyFrom(flat any) error {
	slice, ok := flat.([]T)
	if !ok {
		return errors.Errorf("flat data type %T does not match tensor type %T", flat, slice)
	}
	copy(t.tensor.GetData(), slice)
	return nil
}

// float16 and bfloat16 converters

func float16ToFloat32(src []float16.Float16) []float32 {
	dst := make([]float32, len(src))
	for i, v := range src {
		dst[i] = v.Float32()
	}
	return dst
}

func float32ToFloat16(src []float32, dst []float16.Float16) {
	for i, v := range src {
		dst[i] = float16.FromFloat32(v)
	}
}

func bfloat16ToFloat32(src []bfloat16.BFloat16) []float32 {
	dst := make([]float32, len(src))
	for i, v := range src {
		dst[i] = v.Float32()
	}
	return dst
}

func float32ToBFloat16(src []float32, dst []bfloat16.BFloat16) {
	for i, v := range src {
		dst[i] = bfloat16.FromFloat32(v)
	}
}

type float16Tensor struct {
	tensor *ort.Tensor[float32]
}

func (t *float16Tensor) GetShape() ort.Shape {
	return t.tensor.GetShape()
}

func (t *float16Tensor) Destroy() error {
	return t.tensor.Destroy()
}

func (t *float16Tensor) Value() ort.Value {
	return t.tensor
}

func (t *float16Tensor) ToFlatData(flat any) error {
	slice, ok := flat.([]float16.Float16)
	if !ok {
		return errors.Errorf("flat data type %T does not match float16 slice", flat)
	}
	float32ToFloat16(t.tensor.GetData(), slice)
	return nil
}

func (t *float16Tensor) CopyFrom(flat any) error {
	slice, ok := flat.([]float16.Float16)
	if !ok {
		return errors.Errorf("flat data type %T does not match float16 slice", flat)
	}
	f32 := float16ToFloat32(slice)
	copy(t.tensor.GetData(), f32)
	return nil
}

type bfloat16Tensor struct {
	tensor *ort.Tensor[float32]
}

func (t *bfloat16Tensor) GetShape() ort.Shape {
	return t.tensor.GetShape()
}

func (t *bfloat16Tensor) Destroy() error {
	return t.tensor.Destroy()
}

func (t *bfloat16Tensor) Value() ort.Value {
	return t.tensor
}

func (t *bfloat16Tensor) ToFlatData(flat any) error {
	slice, ok := flat.([]bfloat16.BFloat16)
	if !ok {
		return errors.Errorf("flat data type %T does not match bfloat16 slice", flat)
	}
	float32ToBFloat16(t.tensor.GetData(), slice)
	return nil
}

func (t *bfloat16Tensor) CopyFrom(flat any) error {
	slice, ok := flat.([]bfloat16.BFloat16)
	if !ok {
		return errors.Errorf("flat data type %T does not match bfloat16 slice", flat)
	}
	f32 := bfloat16ToFloat32(slice)
	copy(t.tensor.GetData(), f32)
	return nil
}

func toInt64s(dims []int) []int64 {
	if len(dims) == 0 {
		return []int64{1}
	}
	res := make([]int64, len(dims))
	for i, v := range dims {
		res[i] = int64(v)
	}
	return res
}

func newOrtTensorWrapper(shape shapes.Shape, flat any) (ortTensorWrapper, error) {
	ortShape := ort.NewShape(toInt64s(shape.Dimensions)...)
	switch f := flat.(type) {
	case []float16.Float16:
		f32 := float16ToFloat32(f)
		t, err := ort.NewTensor(ortShape, f32)
		if err != nil {
			return nil, err
		}
		return &float16Tensor{tensor: t}, nil
	case []bfloat16.BFloat16:
		f32 := bfloat16ToFloat32(f)
		t, err := ort.NewTensor(ortShape, f32)
		if err != nil {
			return nil, err
		}
		return &bfloat16Tensor{tensor: t}, nil
	case []float32:
		t, err := ort.NewTensor(ortShape, f)
		if err != nil {
			return nil, err
		}
		return &typedTensor[float32]{tensor: t}, nil
	case []float64:
		t, err := ort.NewTensor(ortShape, f)
		if err != nil {
			return nil, err
		}
		return &typedTensor[float64]{tensor: t}, nil
	case []int32:
		t, err := ort.NewTensor(ortShape, f)
		if err != nil {
			return nil, err
		}
		return &typedTensor[int32]{tensor: t}, nil
	case []int64:
		t, err := ort.NewTensor(ortShape, f)
		if err != nil {
			return nil, err
		}
		return &typedTensor[int64]{tensor: t}, nil
	case []bool:
		t, err := ort.NewTensor(ortShape, f)
		if err != nil {
			return nil, err
		}
		return &typedTensor[bool]{tensor: t}, nil
	case []int8:
		t, err := ort.NewTensor(ortShape, f)
		if err != nil {
			return nil, err
		}
		return &typedTensor[int8]{tensor: t}, nil
	case []uint8:
		t, err := ort.NewTensor(ortShape, f)
		if err != nil {
			return nil, err
		}
		return &typedTensor[uint8]{tensor: t}, nil
	case []int16:
		t, err := ort.NewTensor(ortShape, f)
		if err != nil {
			return nil, err
		}
		return &typedTensor[int16]{tensor: t}, nil
	case []uint16:
		t, err := ort.NewTensor(ortShape, f)
		if err != nil {
			return nil, err
		}
		return &typedTensor[uint16]{tensor: t}, nil
	case []uint32:
		t, err := ort.NewTensor(ortShape, f)
		if err != nil {
			return nil, err
		}
		return &typedTensor[uint32]{tensor: t}, nil
	case []uint64:
		t, err := ort.NewTensor(ortShape, f)
		if err != nil {
			return nil, err
		}
		return &typedTensor[uint64]{tensor: t}, nil
	default:
		return nil, errors.Errorf("unsupported type %T for onnxruntime tensor creation", flat)
	}
}

func newEmptyOrtTensorWrapper(shape shapes.Shape) (ortTensorWrapper, error) {
	ortShape := ort.NewShape(toInt64s(shape.Dimensions)...)
	switch shape.DType {
	case dtypes.Float16:
		t, err := ort.NewEmptyTensor[float32](ortShape)
		if err != nil {
			return nil, err
		}
		return &float16Tensor{tensor: t}, nil
	case dtypes.BFloat16:
		t, err := ort.NewEmptyTensor[float32](ortShape)
		if err != nil {
			return nil, err
		}
		return &bfloat16Tensor{tensor: t}, nil
	case dtypes.Float32:
		t, err := ort.NewEmptyTensor[float32](ortShape)
		if err != nil {
			return nil, err
		}
		return &typedTensor[float32]{tensor: t}, nil
	case dtypes.Float64:
		t, err := ort.NewEmptyTensor[float64](ortShape)
		if err != nil {
			return nil, err
		}
		return &typedTensor[float64]{tensor: t}, nil
	case dtypes.Int32:
		t, err := ort.NewEmptyTensor[int32](ortShape)
		if err != nil {
			return nil, err
		}
		return &typedTensor[int32]{tensor: t}, nil
	case dtypes.Int64:
		t, err := ort.NewEmptyTensor[int64](ortShape)
		if err != nil {
			return nil, err
		}
		return &typedTensor[int64]{tensor: t}, nil
	case dtypes.Bool:
		t, err := ort.NewEmptyTensor[bool](ortShape)
		if err != nil {
			return nil, err
		}
		return &typedTensor[bool]{tensor: t}, nil
	case dtypes.Int8:
		t, err := ort.NewEmptyTensor[int8](ortShape)
		if err != nil {
			return nil, err
		}
		return &typedTensor[int8]{tensor: t}, nil
	case dtypes.Uint8:
		t, err := ort.NewEmptyTensor[uint8](ortShape)
		if err != nil {
			return nil, err
		}
		return &typedTensor[uint8]{tensor: t}, nil
	case dtypes.Int16:
		t, err := ort.NewEmptyTensor[int16](ortShape)
		if err != nil {
			return nil, err
		}
		return &typedTensor[int16]{tensor: t}, nil
	case dtypes.Uint16:
		t, err := ort.NewEmptyTensor[uint16](ortShape)
		if err != nil {
			return nil, err
		}
		return &typedTensor[uint16]{tensor: t}, nil
	case dtypes.Uint32:
		t, err := ort.NewEmptyTensor[uint32](ortShape)
		if err != nil {
			return nil, err
		}
		return &typedTensor[uint32]{tensor: t}, nil
	case dtypes.Uint64:
		t, err := ort.NewEmptyTensor[uint64](ortShape)
		if err != nil {
			return nil, err
		}
		return &typedTensor[uint64]{tensor: t}, nil
	default:
		return nil, errors.Errorf("unsupported DType %s for empty onnxruntime tensor allocation", shape.DType)
	}
}

// Buffer implements compute.Buffer for ONNX Runtime.
type Buffer struct {
	backend *Backend
	wrapper ortTensorWrapper
	shape   shapes.Shape
	device  compute.DeviceNum
}

var _ compute.Buffer = (*Buffer)(nil)

func (b *Buffer) Backend() compute.Backend {
	return b.backend
}

func (b *Buffer) Finalize() error {
	if b.wrapper != nil {
		err := b.wrapper.Destroy()
		b.wrapper = nil
		if err != nil {
			return errors.Wrap(err, "failed to destroy onnxruntime tensor")
		}
	}
	return nil
}

func (b *Buffer) Shape() (shapes.Shape, error) {
	return b.shape, nil
}

func (b *Buffer) DeviceNum() (compute.DeviceNum, error) {
	return b.device, nil
}

func (b *Buffer) ToFlatData(flat any) error {
	if b.wrapper == nil {
		return errors.New("cannot read from finalized buffer")
	}
	return b.wrapper.ToFlatData(flat)
}

func (b *Buffer) Data() (flat any, err error) {
	return nil, errors.New("shared buffers not supported")
}

func (b *Buffer) CopyToDevice(deviceNum compute.DeviceNum) (compute.Buffer, error) {
	if b.wrapper == nil {
		return nil, errors.New("cannot copy finalized buffer")
	}
	newWrapper, err := newEmptyOrtTensorWrapper(b.shape)
	if err != nil {
		return nil, err
	}
	flatSlice := dtypes.MakeAnySlice(b.shape.DType, b.shape.Size())
	err = b.wrapper.ToFlatData(flatSlice)
	if err != nil {
		newWrapper.Destroy()
		return nil, err
	}
	err = newWrapper.CopyFrom(flatSlice)
	if err != nil {
		newWrapper.Destroy()
		return nil, err
	}
	return &Buffer{
		backend: b.backend,
		wrapper: newWrapper,
		shape:   b.shape,
		device:  deviceNum,
	}, nil
}

func wrapOrtValue(val ort.Value, shape shapes.Shape) (ortTensorWrapper, error) {
	switch shape.DType {
	case dtypes.Float16:
		t, ok := val.(*ort.Tensor[float32])
		if !ok {
			return nil, errors.Errorf("expected *ort.Tensor[float32], got %T", val)
		}
		return &float16Tensor{tensor: t}, nil
	case dtypes.BFloat16:
		t, ok := val.(*ort.Tensor[float32])
		if !ok {
			return nil, errors.Errorf("expected *ort.Tensor[float32], got %T", val)
		}
		return &bfloat16Tensor{tensor: t}, nil
	case dtypes.Float32:
		t, ok := val.(*ort.Tensor[float32])
		if !ok {
			return nil, errors.Errorf("expected *ort.Tensor[float32], got %T", val)
		}
		return &typedTensor[float32]{tensor: t}, nil
	case dtypes.Float64:
		t, ok := val.(*ort.Tensor[float64])
		if !ok {
			return nil, errors.Errorf("expected *ort.Tensor[float64], got %T", val)
		}
		return &typedTensor[float64]{tensor: t}, nil
	case dtypes.Int32:
		t, ok := val.(*ort.Tensor[int32])
		if !ok {
			return nil, errors.Errorf("expected *ort.Tensor[int32], got %T", val)
		}
		return &typedTensor[int32]{tensor: t}, nil
	case dtypes.Int64:
		t, ok := val.(*ort.Tensor[int64])
		if !ok {
			return nil, errors.Errorf("expected *ort.Tensor[int64], got %T", val)
		}
		return &typedTensor[int64]{tensor: t}, nil
	case dtypes.Bool:
		t, ok := val.(*ort.Tensor[bool])
		if !ok {
			return nil, errors.Errorf("expected *ort.Tensor[bool], got %T", val)
		}
		return &typedTensor[bool]{tensor: t}, nil
	case dtypes.Int8:
		t, ok := val.(*ort.Tensor[int8])
		if !ok {
			return nil, errors.Errorf("expected *ort.Tensor[int8], got %T", val)
		}
		return &typedTensor[int8]{tensor: t}, nil
	case dtypes.Uint8:
		t, ok := val.(*ort.Tensor[uint8])
		if !ok {
			return nil, errors.Errorf("expected *ort.Tensor[uint8], got %T", val)
		}
		return &typedTensor[uint8]{tensor: t}, nil
	case dtypes.Int16:
		t, ok := val.(*ort.Tensor[int16])
		if !ok {
			return nil, errors.Errorf("expected *ort.Tensor[int16], got %T", val)
		}
		return &typedTensor[int16]{tensor: t}, nil
	case dtypes.Uint16:
		t, ok := val.(*ort.Tensor[uint16])
		if !ok {
			return nil, errors.Errorf("expected *ort.Tensor[uint16], got %T", val)
		}
		return &typedTensor[uint16]{tensor: t}, nil
	case dtypes.Uint32:
		t, ok := val.(*ort.Tensor[uint32])
		if !ok {
			return nil, errors.Errorf("expected *ort.Tensor[uint32], got %T", val)
		}
		return &typedTensor[uint32]{tensor: t}, nil
	case dtypes.Uint64:
		t, ok := val.(*ort.Tensor[uint64])
		if !ok {
			return nil, errors.Errorf("expected *ort.Tensor[uint64], got %T", val)
		}
		return &typedTensor[uint64]{tensor: t}, nil
	default:
		return nil, errors.Errorf("unsupported shape DType %s for wrapOrtValue", shape.DType)
	}
}
