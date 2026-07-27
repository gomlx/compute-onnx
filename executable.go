// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxruntime

import (
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	ort "github.com/gomlx/compute-onnx/internal/ort"
	"github.com/pkg/errors"
)

// cudaExecCtx holds per-execution CUDA state. Pooled for parallel-safe execution.
type cudaExecCtx struct {
	ioBinding   *ort.IoBinding
	cudaMemInfo *ort.MemoryInfo
}

// cudaExecPool manages a pool of cudaExecCtx with proper lifecycle tracking.
type cudaExecPool struct {
	mu      sync.Mutex
	free    []*cudaExecCtx
	all     []*cudaExecCtx
	newFunc func() (*cudaExecCtx, error)
}

func (p *cudaExecPool) get() (*cudaExecCtx, error) {
	p.mu.Lock()
	if n := len(p.free); n > 0 {
		ctx := p.free[n-1]
		p.free = p.free[:n-1]
		p.mu.Unlock()
		return ctx, nil
	}
	p.mu.Unlock()
	ctx, err := p.newFunc()
	if err != nil {
		return nil, err
	}
	p.mu.Lock()
	p.all = append(p.all, ctx)
	p.mu.Unlock()
	return ctx, nil
}

func (p *cudaExecPool) put(ctx *cudaExecCtx) {
	p.mu.Lock()
	p.free = append(p.free, ctx)
	p.mu.Unlock()
}

func (p *cudaExecPool) destroyAll() {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, ctx := range p.all {
		ctx.ioBinding.Destroy()
		_ = ctx.cudaMemInfo.Destroy()
	}
	p.all = nil
	p.free = nil
}

type Executable struct {
	backend          *Backend
	session          *ort.DynamicAdvancedSession
	inputNames       []string
	inputShapes      []shapes.Shape
	outputNames      []string
	outputShapes     []shapes.Shape
	reusableWrappers []ortTensorWrapper

	// Pre-allocated slices reused across Execute calls (CPU path only, single-threaded).
	cachedOrtInputs  []ort.Value
	cachedOutWraps   []ortTensorWrapper
	cachedOrtOutputs []ort.Value

	// CUDA execution context pool — one context per concurrent execution.
	cudaCtxPool cudaExecPool

	// Mutex to protect reusableWrappers for concurrent buffer finalization.
	mu sync.Mutex
}

var _ compute.Executable = (*Executable)(nil)

func newExecutable(backend *Backend, session *ort.DynamicAdvancedSession,
	inputNames []string, inputShapes []shapes.Shape,
	outputNames []string, outputShapes []shapes.Shape) *Executable {

	nInputs := len(inputNames)
	nOutputs := len(outputShapes)

	return &Executable{
		backend:          backend,
		session:          session,
		inputNames:       inputNames,
		inputShapes:      inputShapes,
		outputNames:      outputNames,
		outputShapes:     outputShapes,
		cachedOrtInputs:  make([]ort.Value, nInputs),
		cachedOutWraps:   make([]ortTensorWrapper, nOutputs),
		cachedOrtOutputs: make([]ort.Value, nOutputs),
	}
}

func (e *Executable) Finalize() {
	e.cudaCtxPool.destroyAll()
	if e.session != nil {
		_ = e.session.Destroy()
		e.session = nil
	}
	e.mu.Lock()
	for _, w := range e.reusableWrappers {
		if w != nil {
			_ = w.Destroy()
		}
	}
	e.reusableWrappers = nil
	e.mu.Unlock()
}

func (e *Executable) Inputs() (names []string, inputShapes []shapes.Shape) {
	return e.inputNames, e.inputShapes
}

func (e *Executable) Outputs() (outputShapes []shapes.Shape) {
	return e.outputShapes
}

func wrapperMatchesShape(w ortTensorWrapper, sh shapes.Shape) bool {
	if w.GetDType() != sh.DType {
		return false
	}
	rwShape := w.GetShape()
	if len(rwShape) != len(sh.Dimensions) {
		return false
	}
	for k, dim := range rwShape {
		if dim != int64(sh.Dimensions[k]) {
			return false
		}
	}
	return true
}

func (e *Executable) matchesAnyOutput(dtype dtypes.DType, shape ort.Shape) bool {
	for _, outShape := range e.outputShapes {
		if outShape.DType == dtype && len(shape) == len(outShape.Dimensions) {
			match := true
			for k, dim := range shape {
				if dim != int64(outShape.Dimensions[k]) {
					match = false
					break
				}
			}
			if match {
				return true
			}
		}
	}
	return false
}

// recycleWrapper returns a wrapper to the reusable pool if it matches any output shape.
// If not recyclable or pool is full, the wrapper is destroyed immediately.
// Thread-safe.
func (e *Executable) recycleWrapper(w ortTensorWrapper) {
	if w == nil {
		return
	}

	// Don't recycle on CUDA — buffer management is different.
	if e.backend.cuda {
		_ = w.Destroy()
		return
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	// Don't recycle if executable is finalized.
	if e.session == nil {
		_ = w.Destroy()
		return
	}

	if e.matchesAnyOutput(w.GetDType(), w.GetShape()) && len(e.reusableWrappers) < len(e.outputShapes)*2 {
		e.reusableWrappers = append(e.reusableWrappers, w)
	} else {
		_ = w.Destroy()
	}
}

func (e *Executable) Execute(inputs []compute.Buffer, donate []bool, defaultDevice compute.DeviceNum) ([]compute.Buffer, error) {
	if e.session == nil {
		return nil, errors.New("cannot execute finalized executable")
	}

	isDummyInput := len(e.inputNames) == 1 && e.inputNames[0] == "dummy_input"
	expectedInputs := len(e.inputNames)
	if isDummyInput {
		expectedInputs = 0
	}

	if len(inputs) != expectedInputs {
		return nil, errors.Errorf("expected %d inputs, got %d", expectedInputs, len(inputs))
	}

	if e.backend.cuda {
		// CUDA path: build a local ortInputs slice (e.cachedOrtInputs is not safe for concurrent calls).
		ortInputs := make([]ort.Value, len(e.inputNames))
		var dummyWrapper ortTensorWrapper
		if isDummyInput {
			wrapper, err := newOrtTensorWrapper(e.inputShapes[0], []int32{0})
			if err != nil {
				return nil, err
			}
			dummyWrapper = wrapper
			ortInputs[0] = wrapper.Value()
		} else {
			for i, inp := range inputs {
				buf, ok := inp.(*Buffer)
				if !ok {
					return nil, errors.Errorf("input %d is not a valid onnxruntime buffer", i)
				}
				if buf.wrapper == nil {
					return nil, errors.Errorf("input %d is finalized", i)
				}
				ortInputs[i] = buf.wrapper.Value()
			}
		}
		result, err := e.executeCUDA(ortInputs, inputs, donate, defaultDevice)
		if dummyWrapper != nil {
			_ = dummyWrapper.Destroy()
		}
		return result, err
	}

	// CPU path: populate shared cachedOrtInputs (single-threaded).
	var dummyInputWrapper ortTensorWrapper
	if isDummyInput {
		wrapper, err := newOrtTensorWrapper(e.inputShapes[0], []int32{0})
		if err != nil {
			return nil, err
		}
		dummyInputWrapper = wrapper
		e.cachedOrtInputs[0] = wrapper.Value()
	} else {
		for i, inp := range inputs {
			buf, ok := inp.(*Buffer)
			if !ok {
				return nil, errors.Errorf("input %d is not a valid onnxruntime buffer", i)
			}
			if buf.wrapper == nil {
				return nil, errors.Errorf("input %d is finalized", i)
			}
			e.cachedOrtInputs[i] = buf.wrapper.Value()
		}
	}

	if dummyInputWrapper != nil {
		defer func() {
			_ = dummyInputWrapper.Destroy()
		}()
	}

	return e.executeCPU(inputs, donate, defaultDevice)
}

// executeCUDA uses IoBinding to keep data on GPU between executions.
// Uses a pooled execution context for parallel safety.
func (e *Executable) executeCUDA(ortInputs []ort.Value, inputs []compute.Buffer, donate []bool, defaultDevice compute.DeviceNum) ([]compute.Buffer, error) {
	// Initialize the pool's factory function lazily (only once).
	e.cudaCtxPool.mu.Lock()
	if e.cudaCtxPool.newFunc == nil {
		e.cudaCtxPool.newFunc = func() (*cudaExecCtx, error) {
			ioBinding, err := e.session.CreateIoBinding()
			if err != nil {
				return nil, errors.Wrap(err, "failed to create IoBinding")
			}
			cudaMemInfo, err := ort.NewCUDAMemoryInfo()
			if err != nil {
				ioBinding.Destroy()
				return nil, errors.Wrap(err, "failed to create CUDA MemoryInfo")
			}
			return &cudaExecCtx{ioBinding: ioBinding, cudaMemInfo: cudaMemInfo}, nil
		}
	}
	e.cudaCtxPool.mu.Unlock()

	ctx, err := e.cudaCtxPool.get()
	if err != nil {
		return nil, err
	}

	binding := ctx.ioBinding
	cInputNames := e.session.CInputNames()
	cOutputNames := e.session.COutputNames()

	// 1. Bind inputs.
	for i := range ortInputs {
		if err := binding.BindInput(cInputNames[i], ortInputs[i]); err != nil {
			e.cudaCtxPool.put(ctx)
			return nil, errors.Wrapf(err, "failed to bind input %d", i)
		}
	}

	// 2. Bind outputs — let ORT allocate on GPU.
	for i := range e.outputShapes {
		if err := binding.BindOutputToDevice(cOutputNames[i], ctx.cudaMemInfo); err != nil {
			e.cudaCtxPool.put(ctx)
			return nil, errors.Wrapf(err, "failed to bind output %d to CUDA", i)
		}
	}

	// 3. Execute.
	if err := binding.RunWithBinding(); err != nil {
		e.cudaCtxPool.put(ctx)
		return nil, errors.Wrap(err, "onnxruntime CUDA execution failed")
	}

	// 4. Retrieve output OrtValues (GPU-resident, caller owns them).
	outputValues, err := binding.GetBoundOutputValues()
	if err != nil {
		e.cudaCtxPool.put(ctx)
		return nil, errors.Wrap(err, "failed to get bound output values")
	}

	// Return context to pool — output OrtValues are now owned by the caller
	// via GetBoundOutputValues, independent of the binding's lifecycle.
	e.cudaCtxPool.put(ctx)

	// 5. Wrap outputs in CUDA Buffers.
	outBuffers := make([]compute.Buffer, len(e.outputShapes))
	for i, sh := range e.outputShapes {
		ortShape := ort.NewShape(toInt64s(sh.Dimensions)...)
		val := ort.WrapRawOrtValue(outputValues[i])
		outBuffers[i] = &Buffer{
			backend: e.backend,
			wrapper: &gpuTensorWrapper{
				val:   val,
				shape: ortShape,
				dtype: sh.DType,
			},
			shape:  sh,
			device: defaultDevice,
			isCUDA: true,
		}
	}

	// 6. Finalize all donated inputs.
	for i, inp := range inputs {
		if i < len(donate) && donate[i] {
			buf := inp.(*Buffer)
			if buf.wrapper != nil {
				_ = buf.wrapper.Destroy()
				buf.wrapper = nil
			}
		}
	}

	return outBuffers, nil
}

// executeCPU is the optimized execution path for CPU with pre-allocated outputs and recycling.
func (e *Executable) executeCPU(inputs []compute.Buffer, donate []bool, defaultDevice compute.DeviceNum) ([]compute.Buffer, error) {
	outWrappers := e.cachedOutWraps
	ortOutputs := e.cachedOrtOutputs

	e.mu.Lock()
	for i, sh := range e.outputShapes {
		matchedIdx := -1
		for idx, rw := range e.reusableWrappers {
			if wrapperMatchesShape(rw, sh) {
				matchedIdx = idx
				break
			}
		}

		if matchedIdx >= 0 {
			outWrappers[i] = e.reusableWrappers[matchedIdx]
			// Swap-remove for O(1) deletion.
			last := len(e.reusableWrappers) - 1
			e.reusableWrappers[matchedIdx] = e.reusableWrappers[last]
			e.reusableWrappers[last] = nil
			e.reusableWrappers = e.reusableWrappers[:last]
		} else {
			outWrappers[i] = nil // mark as needing allocation
		}
	}
	e.mu.Unlock()

	// Allocate any output wrappers that weren't found in pool (outside lock).
	for i, sh := range e.outputShapes {
		if outWrappers[i] == nil {
			var err error
			outWrappers[i], err = newEmptyOrtTensorWrapper(sh)
			if err != nil {
				for _, w := range outWrappers {
					if w != nil {
						_ = w.Destroy()
					}
				}
				return nil, err
			}
		}
		ortOutputs[i] = outWrappers[i].Value()
	}

	err := e.session.Run(e.cachedOrtInputs, ortOutputs)
	if err != nil {
		for _, w := range outWrappers {
			if w != nil {
				_ = w.Destroy()
			}
		}
		return nil, errors.Wrap(err, "onnxruntime execution failed")
	}

	// Construct output buffers wrapping our C-heap memory zero-copy.
	// Must allocate fresh slice each call since it escapes to the caller.
	outBuffers := make([]compute.Buffer, len(e.outputShapes))
	for i, sh := range e.outputShapes {
		outBuffers[i] = &Buffer{
			backend:    e.backend,
			wrapper:    outWrappers[i],
			shape:      sh,
			device:     defaultDevice,
			isShared:   true,
			executable: e, // back-pointer for recycling on Finalize
		}
		outWrappers[i] = nil // ownership transferred to Buffer
	}

	// Donated input wrappers are destroyed, not recycled. While GoMLX doesn't
	// access them after Execute(), their data layout may not match what ORT
	// expects for pre-allocated outputs. Only output wrappers (returned via
	// Buffer.Finalize) are known-compatible with the pre-allocated output path.
	for i, inp := range inputs {
		if i < len(donate) && donate[i] {
			buf := inp.(*Buffer)
			w := buf.wrapper
			buf.wrapper = nil

			if w != nil {
				_ = w.Destroy()
			}
		}
	}

	return outBuffers, nil
}
