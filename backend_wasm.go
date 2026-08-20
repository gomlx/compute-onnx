// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build js && wasm

package onnxbackend

import (
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute-onnx/internal/engine/web"
	onnx "github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// Type aliases for web engine types.
type Executable = web.Executable
type Buffer = web.Buffer

const SaveOnFailureEnv = "GOMLX_ONNX_SAVE_ON_FAILURE"

// IsSupportedPlatform returns true when compiled for js/wasm.
func IsSupportedPlatform() bool {
	return true
}

func parseConfig(config string) (ep string, err error) {
	config = strings.TrimSpace(config)
	if config == "" {
		if web.HasWebGPU() {
			return "webgpu", nil
		}
		return "wasm", nil
	}
	if strings.Contains(config, ":") || strings.EqualFold(config, "onnx") || strings.EqualFold(config, "onnxruntime") {
		parsed, errEnv := ParseGOMLXBackendEnv(config)
		if errEnv == nil {
			config = parsed
		}
	}
	parts := strings.Split(config, ",")
	for _, part := range parts {
		part = strings.ToLower(strings.TrimSpace(part))
		if part == "" {
			continue
		}
		if part == "webgpu" || part == "gpu" {
			ep = "webgpu"
		} else if part == "wasm" || part == "cpu" {
			ep = "wasm"
		} else if part == "webgl" {
			ep = "webgl"
		} else if part == "webnn" {
			ep = "webnn"
		} else {
			return "", errors.Errorf("unknown web backend option: %q (expected \"webgpu\", \"wasm\", \"webgl\", \"webnn\", or \"cpu\"/\"gpu\")", part)
		}
	}
	if ep == "" {
		if web.HasWebGPU() {
			ep = "webgpu"
		} else {
			ep = "wasm"
		}
	}
	return ep, nil
}

// New creates a new ONNX Runtime Web backend instance.
func New(config string) (compute.Backend, error) {
	ep, err := parseConfig(config)
	if err != nil {
		return nil, err
	}

	if ep == "webgpu" && !web.HasWebGPU() {
		return nil, errors.New("WebGPU is not available or failed to acquire a GPU adapter in the current browser environment (navigator.gpu is unavailable or requestAdapter() failed; in Chrome, this may require running with GPU hardware acceleration or enabling --enable-unsafe-webgpu)")
	}

	if ep == "webnn" && !web.HasWebNN() {
		return nil, errors.New("WebNN is not available in the current browser environment (navigator.ml is undefined); WebNN is experimental in most browsers and typically requires enabling a browser flag such as chrome://flags/#web-machine-learning-neural-network)")
	}

	if err := web.EnsureORTLoaded(); err != nil {
		return nil, errors.Wrap(err, "failed to initialize onnxruntime-web")
	}

	return &Backend{
		config:            config,
		executionProvider: ep,
	}, nil
}

func (b *Backend) createExecutable(modelBytes []byte, inputNames []string, inputShapes []shapes.Shape,
	outputNames []string, outputShapes []shapes.Shape, modelProto *onnx.ModelProto) (compute.Executable, error) {

	session, err := web.CreateSession(modelBytes, b.executionProvider)
	if err != nil {
		return nil, err
	}
	var savedModelProto *onnx.ModelProto
	if b.keepModelProto {
		savedModelProto = modelProto
	}
	return web.NewExecutable(b, session, inputNames, inputShapes, outputNames, outputShapes, savedModelProto), nil
}

func (b *Backend) BufferFromFlatData(deviceNum compute.DeviceNum, flat any, shape shapes.Shape) (compute.Buffer, error) {
	typedArray, err := web.ConvertSliceToTypedArray(flat, shape.DType)
	if err != nil {
		return nil, err
	}
	jsTensor, err := web.CreateJSTensor(shape.DType, shape.Dimensions, typedArray)
	if err != nil {
		return nil, err
	}
	wrapper := web.NewWebTensorWrapper(jsTensor, shape)
	return web.NewBuffer(b, wrapper, shape, deviceNum, false, nil), nil
}

func (b *Backend) HasSharedBuffers() bool {
	return false
}

func (b *Backend) NewSharedBuffer(deviceNum compute.DeviceNum, shape shapes.Shape) (compute.Buffer, any, error) {
	return nil, nil, errors.New("shared buffers are not supported in web/wasm backend")
}
