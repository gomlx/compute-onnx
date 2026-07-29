// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"github.com/gomlx/compute"
	onnx "github.com/gomlx/compute-onnx/internal/protos"
	"github.com/gomlx/compute/notimplemented"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

type Node struct {
	name        string
	opType      string
	domain      string
	inputs      []*Node
	outputNames []string
	shape       shapes.Shape
	flatValue   any // used if opType == "Constant"
	attributes  []*onnx.AttributeProto
}

type Builder struct {
	notimplemented.Builder
	name    string
	backend *Backend
	mainFn  *Function
	funcs   map[string]*Function
}

var _ compute.Builder = (*Builder)(nil)

func NewBuilder(name string, backend *Backend) *Builder {
	b := &Builder{
		Builder: notimplemented.Builder{
			ErrFn: func(op compute.OpType) error {
				return errors.Wrapf(compute.ErrNotImplemented, "%s (%d) not implemented for ONNX Runtime backend", op, op)
			},
		},
		name:    name,
		backend: backend,
		funcs:   make(map[string]*Function),
	}
	b.mainFn = NewFunction(compute.MainName, b)
	b.funcs[compute.MainName] = b.mainFn
	return b
}

func (b *Builder) Name() string {
	return b.name
}

func (b *Builder) Main() compute.Function {
	return b.mainFn
}

func (b *Builder) NewFunction(name string) (compute.Function, error) {
	if _, ok := b.funcs[name]; ok {
		return nil, errors.Errorf("function %q already defined in builder %q", name, b.name)
	}
	f := NewFunction(name, b)
	b.funcs[name] = f
	return f, nil
}

func (b *Builder) OpShape(op compute.Value) (shapes.Shape, error) {
	node, ok := op.(*Node)
	if !ok {
		return shapes.Invalid(), errors.New("value is not a valid onnxruntime node")
	}
	return node.shape, nil
}

func (b *Builder) DeviceAssignment(devices ...compute.DeviceNum) error {
	// Only device 0 is supported
	if len(devices) > 1 || (len(devices) == 1 && devices[0] != 0) {
		return errors.Wrap(compute.ErrNotImplemented, "only device 0 is supported by onnxruntime backend")
	}
	return nil
}
