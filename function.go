// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/gomlx/compute"
	onnx "github.com/gomlx/compute-onnx/internal/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/notimplemented"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

type Function struct {
	notimplemented.Function
	name      string
	builder   *Builder
	parent    *Function
	params    []*Node
	nodes     []*Node
	returns   []*Node
	nodeCount int
	nodeCache map[string]*Node
}

var _ compute.Function = (*Function)(nil)

func NewFunction(name string, builder *Builder) *Function {
	return &Function{
		Function: notimplemented.Function{
			ErrFn: func(op compute.OpType) error {
				return errors.Wrapf(compute.ErrNotImplemented, "%s (%d) not implemented for ONNX Runtime backend", op, op)
			},
		},
		name:      name,
		builder:   builder,
		nodeCache: make(map[string]*Node),
	}
}

func formatAttribute(attr *onnx.AttributeProto) string {
	if attr == nil {
		return ""
	}
	return fmt.Sprintf("%s:%d:i=%d:f=%f:s=%s:is=%v:fs=%v;",
		attr.Name, attr.Type, attr.I, attr.F, attr.S, attr.Ints, attr.Floats)
}

func (f *Function) nodeCacheKey(node *Node) string {
	var sb strings.Builder
	sb.WriteString(node.opType)
	sb.WriteByte('|')
	sb.WriteString(node.domain)
	sb.WriteByte('|')
	sb.WriteString(node.shape.String())
	sb.WriteByte('|')
	for _, inp := range node.inputs {
		fmt.Fprintf(&sb, "%p,", inp)
	}
	sb.WriteByte('|')
	for _, attr := range node.attributes {
		sb.WriteString(formatAttribute(attr))
	}
	sb.WriteByte('|')
	if node.opType == "Constant" && node.flatValue != nil {
		raw := dtypes.UnsafeByteSliceFromAny(node.flatValue)
		sb.WriteString(string(raw))
	}
	return sb.String()
}

func (f *Function) addNode(node *Node) *Node {
	if f.nodeCache == nil {
		f.nodeCache = make(map[string]*Node)
	}
	key := f.nodeCacheKey(node)
	if cached, ok := f.nodeCache[key]; ok {
		return cached
	}

	f.nodeCount++
	if node.name == "" {
		if node.opType == "Constant" {
			node.name = fmt.Sprintf("const_%d", f.nodeCount)
		} else {
			node.name = fmt.Sprintf("node_%d", f.nodeCount)
		}
	}
	f.nodes = append(f.nodes, node)
	f.nodeCache[key] = node
	return node
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

func (f *Function) Constant(flat any, dims ...int) (compute.Value, error) {
	valType := reflect.TypeOf(flat)
	if valType.Kind() != reflect.Slice && valType.Kind() != reflect.Array {
		return nil, errors.Errorf("Constant expects flat to be a slice or array, got %T", flat)
	}
	dtype := dtypes.FromGoType(valType.Elem())
	shape := shapes.Make(dtype, dims...)
	node := &Node{
		opType:    "Constant",
		shape:     shape,
		flatValue: flat,
	}
	return f.addNode(node), nil
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
