// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxruntime

import (
	"github.com/gomlx/compute"
)

func (f *Function) Add(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeAdd, "Add", lhs, rhs)
}

func (f *Function) Sub(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeSub, "Sub", lhs, rhs)
}

func (f *Function) Mul(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeMul, "Mul", lhs, rhs)
}

func (f *Function) Div(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeDiv, "Div", lhs, rhs)
}

func (f *Function) Max(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeMax, "Max", lhs, rhs)
}

func (f *Function) Min(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeMin, "Min", lhs, rhs)
}

func (f *Function) Pow(lhs, rhs compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypePow, "Pow", lhs, rhs)
}
