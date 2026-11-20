// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build js && wasm

package web

import (
	"syscall/js"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xsync"
	"github.com/pkg/errors"
)

// WebTensorWrapper wraps a JavaScript ort.Tensor.
type WebTensorWrapper struct {
	jsTensor js.Value
	shape    shapes.Shape
}

// NewWebTensorWrapper creates a wrapper from an existing ort.Tensor JS object.
func NewWebTensorWrapper(jsTensor js.Value, shape shapes.Shape) *WebTensorWrapper {
	return &WebTensorWrapper{
		jsTensor: jsTensor,
		shape:    shape,
	}
}

func (w *WebTensorWrapper) GetShape() shapes.Shape {
	return w.shape
}

func (w *WebTensorWrapper) GetDType() dtypes.DType {
	return w.shape.DType
}

func (w *WebTensorWrapper) JSTensor() js.Value {
	return w.jsTensor
}

func (w *WebTensorWrapper) Destroy() error {
	w.jsTensor = js.Undefined()
	return nil
}

func (w *WebTensorWrapper) ToFlatData(flat any) error {
	if w.jsTensor.IsUndefined() || w.jsTensor.IsNull() {
		return errors.New("cannot read from destroyed web tensor")
	}
	dataProp := w.jsTensor.Get("data")
	if dataProp.IsUndefined() || dataProp.IsNull() {
		return errors.New("web tensor has no data property")
	}
	return CopyTypedArrayToSlice(dataProp, flat, w.shape.DType)
}

func (w *WebTensorWrapper) CopyFrom(flat any) error {
	typedArray, err := ConvertSliceToTypedArray(flat, w.shape.DType)
	if err != nil {
		return err
	}
	jsTensor, err := CreateJSTensor(w.shape.DType, w.shape.Dimensions, typedArray)
	if err != nil {
		return err
	}
	w.jsTensor = jsTensor
	return nil
}

// Buffer implements [compute.Buffer] for Web / WebAssembly execution.
type Buffer struct {
	backend          compute.Backend
	backendLifecycle *xsync.Lifecycle
	wrapper          *WebTensorWrapper
	shape            shapes.Shape
	device           compute.DeviceNum
	isShared         bool
	executable       *Executable
}

var _ compute.Buffer = (*Buffer)(nil)

// NewBuffer creates a new web Buffer.
func NewBuffer(backend compute.Backend, backendLifecycle *xsync.Lifecycle, wrapper *WebTensorWrapper, shape shapes.Shape, device compute.DeviceNum, isShared bool, exec *Executable) *Buffer {
	shape.AxisNames = nil
	return &Buffer{
		backend:          backend,
		backendLifecycle: backendLifecycle,
		wrapper:          wrapper,
		shape:            shape,
		device:           device,
		isShared:         isShared,
		executable:       exec,
	}
}

func (b *Buffer) enterBackend(op string) (bool, error) {
	if b.backendLifecycle != nil {
		if !b.backendLifecycle.Enter() {
			return false, errors.Wrapf(compute.ErrBackendFinalized, "onnxruntime-web: Buffer.%s", op)
		}
		return true, nil
	}
	if b.backend != nil && b.backend.IsFinalized() {
		return false, errors.Wrapf(compute.ErrBackendFinalized, "onnxruntime-web: Buffer.%s", op)
	}
	return false, nil
}

func (b *Buffer) Backend() compute.Backend {
	return b.backend
}

func (b *Buffer) Finalize() error {
	if b == nil {
		return nil
	}
	if b.wrapper != nil {
		w := b.wrapper
		b.wrapper = nil
		return w.Destroy()
	}
	return nil
}

func (b *Buffer) Shape() (shapes.Shape, error) {
	if b == nil || b.wrapper == nil {
		return shapes.Invalid(), errors.New("cannot query shape of finalized buffer")
	}
	if b.backend != nil && b.backend.IsFinalized() {
		return shapes.Invalid(), errors.Wrap(compute.ErrBackendFinalized, "onnxruntime-web: Buffer.Shape")
	}
	return b.shape, nil
}

func (b *Buffer) DeviceNum() (compute.DeviceNum, error) {
	if b == nil || b.wrapper == nil {
		return 0, errors.New("cannot query device of finalized buffer")
	}
	if b.backend != nil && b.backend.IsFinalized() {
		return 0, errors.Wrap(compute.ErrBackendFinalized, "onnxruntime-web: Buffer.DeviceNum")
	}
	return b.device, nil
}

func (b *Buffer) ToFlatData(flat any) error {
	entered, err := b.enterBackend("ToFlatData")
	if err != nil {
		return err
	}
	if entered {
		defer b.backendLifecycle.Leave()
	}
	if b.wrapper == nil {
		return errors.New("cannot read from finalized buffer")
	}
	return b.wrapper.ToFlatData(flat)
}

func (b *Buffer) Data() (flat any, err error) {
	if b.backend != nil && b.backend.IsFinalized() {
		return nil, errors.Wrap(compute.ErrBackendFinalized, "onnxruntime-web: Buffer.Data")
	}
	return nil, errors.New("direct memory access not supported for WebAssembly JS buffers; use ToFlatData")
}

func (b *Buffer) CopyToDevice(deviceNum compute.DeviceNum) (compute.Buffer, error) {
	entered, err := b.enterBackend("CopyToDevice")
	if err != nil {
		return nil, err
	}
	if entered {
		defer b.backendLifecycle.Leave()
	}
	if b.wrapper == nil {
		return nil, errors.New("cannot copy finalized buffer")
	}
	flatSlice := dtypes.MakeAnySlice(b.shape.DType, b.shape.Size())
	err = b.wrapper.ToFlatData(flatSlice)
	if err != nil {
		return nil, err
	}
	typedArray, err := ConvertSliceToTypedArray(flatSlice, b.shape.DType)
	if err != nil {
		return nil, err
	}
	newJSTensor, err := CreateJSTensor(b.shape.DType, b.shape.Dimensions, typedArray)
	if err != nil {
		return nil, err
	}
	newWrapper := NewWebTensorWrapper(newJSTensor, b.shape)
	return &Buffer{
		backend:          b.backend,
		backendLifecycle: b.backendLifecycle,
		wrapper:          newWrapper,
		shape:            b.shape,
		device:           deviceNum,
		isShared:         false,
	}, nil
}

// Wrapper returns the underlying WebTensorWrapper.
func (b *Buffer) Wrapper() *WebTensorWrapper {
	return b.wrapper
}
