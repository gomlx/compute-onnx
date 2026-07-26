// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxruntime

import (
	"github.com/gomlx/compute"
)

func (f *Function) Abs(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeAbs, "Abs", x)
}

func (f *Function) Neg(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeNeg, "Neg", x)
}

func (f *Function) Ceil(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeCeil, "Ceil", x)
}

func (f *Function) Floor(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeFloor, "Floor", x)
}

func (f *Function) Round(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeRound, "Round", x)
}

func (f *Function) Sqrt(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeSqrt, "Sqrt", x)
}

func (f *Function) Exp(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeExp, "Exp", x)
}

func (f *Function) Log(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeLog, "Log", x)
}

func (f *Function) Cos(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeCos, "Cos", x)
}

func (f *Function) Sin(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeSin, "Sin", x)
}

func (f *Function) Tanh(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeTanh, "Tanh", x)
}

func (f *Function) Logistic(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeLogistic, "Sigmoid", x)
}

func (f *Function) Erf(x compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeErf, "Erf", x)
}
