// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/notimplemented"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

type Function struct {
	notimplemented.Function
	name       string
	builder    *Builder
	parent     *Function
	params     []*Node
	nodes      []*Node
	returns    []*Node
	nodeCount  int
	constCache map[string]*Node
}

var _ compute.Function = (*Function)(nil)

func NewFunction(name string, builder *Builder) *Function {
	return &Function{
		Function: notimplemented.Function{
			ErrFn: func(op compute.OpType) error {
				return errors.Wrapf(compute.ErrNotImplemented, "%s (%d) not implemented for ONNX Runtime backend", op, op)
			},
		},
		name:       name,
		builder:    builder,
		constCache: make(map[string]*Node),
	}
}

func (f *Function) addNode(node *Node) *Node {
	f.nodeCount++
	if node.name == "" {
		if node.opType == "Constant" {
			node.name = fmt.Sprintf("const_%d", f.nodeCount)
		} else {
			node.name = fmt.Sprintf("node_%d", f.nodeCount)
		}
	}
	f.nodes = append(f.nodes, node)
	return node
}

func (f *Function) constCacheKey(shape shapes.Shape, flat any) string {
	var sb strings.Builder
	sb.WriteString(shape.String())
	sb.WriteByte('|')
	if flat != nil {
		raw := dtypes.UnsafeByteSliceFromAny(flat)
		sb.Write(raw)
	}
	return sb.String()
}

func (f *Function) Constant(flat any, dims ...int) (compute.Value, error) {
	valType := reflect.TypeOf(flat)
	if valType.Kind() != reflect.Slice && valType.Kind() != reflect.Array {
		return nil, errors.Errorf("Constant expects flat to be a slice or array, got %T", flat)
	}
	dtype := dtypes.FromGoType(valType.Elem())
	shape := shapes.Make(dtype, dims...)

	key := f.constCacheKey(shape, flat)
	if f.constCache == nil {
		f.constCache = make(map[string]*Node)
	}
	if cached, ok := f.constCache[key]; ok {
		return cached, nil
	}

	node := &Node{
		opType:    "Constant",
		shape:     shape,
		flatValue: flat,
	}
	n := f.addNode(node)
	f.constCache[key] = n
	return n, nil
}

func (f *Function) Name() string {
	return f.name
}

func (f *Function) Parent() compute.Function {
	if f.parent != nil {
		return f.parent
	}
	return nil
}

func (f *Function) Builder() compute.Builder {
	return f.builder
}

func (f *Function) Shape(v compute.Value) (shapes.Shape, error) {
	node, ok := v.(*Node)
	if !ok {
		return shapes.Invalid(), errors.New("value is not a valid onnxruntime node")
	}
	return node.shape, nil
}

func (f *Function) Parameter(name string, shape shapes.Shape, spec *compute.ShardingSpec) (compute.Value, error) {
	if name == "" {
		f.nodeCount++
		name = fmt.Sprintf("param_%d", f.nodeCount)
	}
	node := &Node{
		name:   name,
		opType: "Parameter",
		shape:  shape,
	}
	f.params = append(f.params, node)
	f.nodes = append(f.nodes, node)
	return node, nil
}



func (f *Function) Return(outputs []compute.Value, shardings []*compute.ShardingSpec) error {
	f.returns = make([]*Node, len(outputs))
	for i, output := range outputs {
		node, ok := output.(*Node)
		if !ok {
			return errors.New("return value is not a valid onnxruntime node")
		}
		f.returns[i] = node
	}
	return nil
}

func (f *Function) Identity(x compute.Value) (compute.Value, error) {
	xNode, ok := x.(*Node)
	if !ok {
		return nil, errors.New("identity input is not a valid onnxruntime node")
	}

	node := &Node{
		opType: "Identity",
		inputs: []*Node{xNode},
		shape:  xNode.shape,
	}
	return f.addNode(node), nil
}
