// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"fmt"
	"sync/atomic"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute-onnx/internal/executionprovider"
	"github.com/gomlx/compute/notimplemented"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// CompilerFn is the function type used by Builder to compile a graph into a compute.Executable.
type CompilerFn func(b *Builder) (compute.Executable, error)

// Builder implements [compute.Builder] for building ONNX computation graphs.
type Builder struct {
	notimplemented.Builder
	name              string
	executionProvider executionprovider.Type
	logSeverity       int
	compileFn         CompilerFn
	uniqueAxisNameFn  func(base string) string
	uniqueAxisNameID  atomic.Uint64
	mainFn            *Function
	funcs             map[string]*Function
}

var _ compute.Builder = (*Builder)(nil)

// NewBuilder creates a new graph Builder.
func NewBuilder(name string, compileFn CompilerFn) *Builder {
	b := &Builder{
		Builder: notimplemented.Builder{
			ErrFn: func(op compute.OpType) error {
				return errors.Wrapf(compute.ErrNotImplemented, "%s (%d) not implemented for ONNX Runtime backend", op, op)
			},
		},
		name:        name,
		logSeverity: -1,
		compileFn:   compileFn,
		funcs:       make(map[string]*Function),
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

// MainFunction returns the strongly typed main *Function.
func (b *Builder) MainFunction() *Function {
	return b.mainFn
}

// SetExecutionProvider sets the target execution provider.
func (b *Builder) SetExecutionProvider(ep executionprovider.Type) {
	b.executionProvider = ep
	for _, f := range b.funcs {
		f.executionProvider = ep
	}
}

// ExecutionProvider returns the target execution provider.
func (b *Builder) ExecutionProvider() executionprovider.Type {
	return b.executionProvider
}

// SetLogSeverity sets the log severity level for the builder.
func (b *Builder) SetLogSeverity(severity int) {
	b.logSeverity = severity
}

// LogSeverity returns the log severity level for the builder.
func (b *Builder) LogSeverity() int {
	return b.logSeverity
}

// SetUniqueAxisNameFn configures the callback used to mint unique dynamic axis names.
func (b *Builder) SetUniqueAxisNameFn(fn func(base string) string) {
	b.uniqueAxisNameFn = fn
}

// UniqueAxisName returns a name derived from base that is unique across the backend/builder.
func (b *Builder) UniqueAxisName(base string) string {
	if b != nil && b.uniqueAxisNameFn != nil {
		return b.uniqueAxisNameFn(base)
	}
	if b != nil {
		return fmt.Sprintf("%s_%d", base, b.uniqueAxisNameID.Add(1))
	}
	return fmt.Sprintf("%s_1", base)
}

// Functions returns all registered functions in the builder.
func (b *Builder) Functions() map[string]*Function {
	return b.funcs
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

func (b *Builder) Compile() (compute.Executable, error) {
	if b.compileFn == nil {
		return nil, errors.New("no compiler configured for builder")
	}
	return b.compileFn(b)
}
