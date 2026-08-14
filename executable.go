// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"runtime"
	"sync"
	"time"

	"github.com/gomlx/compute"
	ort "github.com/gomlx/compute-onnx/internal/ort"
	onnx "github.com/gomlx/compute-onnx/internal/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/humanize"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

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

	// Mutex to protect reusableWrappers for concurrent buffer finalization.
	mu sync.Mutex

	modelProto *onnx.ModelProto
}

var _ compute.Executable = (*Executable)(nil)

func newExecutable(backend *Backend, session *ort.DynamicAdvancedSession,
	inputNames []string, inputShapes []shapes.Shape,
	outputNames []string, outputShapes []shapes.Shape,
	modelProto *onnx.ModelProto) *Executable {

	nInputs := len(inputNames)
	nOutputs := len(outputShapes)

	e := &Executable{
		backend:          backend,
		session:          session,
		inputNames:       inputNames,
		inputShapes:      inputShapes,
		outputNames:      outputNames,
		outputShapes:     outputShapes,
		cachedOrtInputs:  make([]ort.Value, nInputs),
		cachedOutWraps:   make([]ortTensorWrapper, nOutputs),
		cachedOrtOutputs: make([]ort.Value, nOutputs),
		modelProto:       modelProto,
	}
	runtime.SetFinalizer(e, (*Executable).Finalize)
	return e
}

// ModelProto returns the ONNX ModelProto struct if Backend.KeepModelProto was enabled during compilation, or nil otherwise.
func (e *Executable) ModelProto() *onnx.ModelProto {
	return e.modelProto
}

func (e *Executable) Finalize() {
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

	defer runtime.KeepAlive(inputs)
	if e.backend.cuda {

		// CUDA path: build local ortInputs slice.
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

	e.mu.Lock()
	defer e.mu.Unlock()

	// Populate shared cachedOrtInputs.
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

// executeCUDA uses IoBinding to run ONNX Runtime models on GPU.
func (e *Executable) executeCUDA(ortInputs []ort.Value, inputs []compute.Buffer, donate []bool, defaultDevice compute.DeviceNum) ([]compute.Buffer, error) {
	defer runtime.KeepAlive(inputs)
	defer runtime.KeepAlive(ortInputs)
	ioBinding, err := e.session.CreateIoBinding()
	if err != nil {
		return nil, errors.Wrap(err, "failed to create IoBinding")
	}
	defer ioBinding.Destroy()

	cudaMemInfo, err := ort.NewCUDAMemoryInfo()
	if err != nil {
		return nil, errors.Wrap(err, "failed to create CUDA MemoryInfo")
	}
	defer cudaMemInfo.Destroy()

	cInputNames := e.session.CInputNames()
	cOutputNames := e.session.COutputNames()

	for i := range ortInputs {
		if err := ioBinding.BindInput(cInputNames[i], ortInputs[i]); err != nil {
			return nil, errors.Wrapf(err, "failed to bind input %d", i)
		}
	}

	for i := range e.outputShapes {
		if err := ioBinding.BindOutputToDevice(cOutputNames[i], cudaMemInfo); err != nil {
			return nil, errors.Wrapf(err, "failed to bind output %d to CUDA", i)
		}
	}

	var start time.Time
	if klog.V(1).Enabled() {
		start = time.Now()
	}
	if klog.V(2).Enabled() {
		klog.Infof("Starting IoBinding.RunWithBinding on cuda (inputs=%d, outputs=%d)...", len(ortInputs), len(e.outputShapes))
	}
	e.mu.Lock()
	err = ioBinding.RunWithBinding()
	e.mu.Unlock()
	if klog.V(1).Enabled() {
		klog.Infof("Execution (CUDA) elapsed time: %s\n", humanize.Duration(time.Since(start)))
	}
	if err != nil {
		return nil, errors.Wrap(err, "onnxruntime CUDA execution failed")
	}

	outputValues, err := ioBinding.GetBoundOutputValues()
	if err != nil {
		return nil, errors.Wrap(err, "failed to get bound output values")
	}

	outBuffers := make([]compute.Buffer, len(e.outputShapes))
	for i, sh := range e.outputShapes {
		val := ort.WrapRawOrtValue(outputValues[i])
		actualShape := sh
		if sh.IsDynamic() {
			boundShape, err := ort.GetOrtValueShape(outputValues[i])
			if err != nil {
				return nil, errors.Wrapf(err, "failed to get shape for output %d", i)
			}
			concreteDims := make([]int, len(boundShape))
			for k, dim := range boundShape {
				concreteDims[k] = int(dim)
			}
			actualShape.Dimensions = concreteDims
		}
		ortShape := ort.NewShape(toInt64s(actualShape.Dimensions)...)
		outBuffers[i] = &Buffer{
			backend: e.backend,
			wrapper: &gpuTensorWrapper{
				val:   val,
				shape: ortShape,
				dtype: actualShape.DType,
			},
			shape:  actualShape,
			device: defaultDevice,
			isCUDA: true,
		}
	}

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

	for i, sh := range e.outputShapes {
		if sh.IsDynamic() {
			outWrappers[i] = nil
			continue
		}
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

	// Allocate any output wrappers that weren't found in pool (outside lock).
	for i, sh := range e.outputShapes {
		if sh.IsDynamic() {
			ortOutputs[i] = nil
			continue
		}
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

	var start time.Time
	if klog.V(1).Enabled() {
		start = time.Now()
	}
	if klog.V(2).Enabled() {
		klog.Infof("Starting session.Run on %s (inputs=%d, outputs=%d)...", e.backend.config, len(e.cachedOrtInputs), len(ortOutputs))
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	err := e.session.Run(e.cachedOrtInputs, ortOutputs)
	if klog.V(1).Enabled() {
		klog.Infof("Execution (%s) elapsed time: %s\n", e.backend.config, humanize.Duration(time.Since(start)))
	}
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
		actualShape := sh
		if sh.IsDynamic() {
			var err error
			outWrappers[i], err = wrapOrtValue(ortOutputs[i], sh)
			if err != nil {
				return nil, err
			}
			actualOrtShape := outWrappers[i].GetShape()
			concreteDims := make([]int, len(actualOrtShape))
			for k, dim := range actualOrtShape {
				concreteDims[k] = int(dim)
			}
			actualShape.Dimensions = concreteDims
		}

		execBackpointer := e
		if e.backend.cuda {
			execBackpointer = nil // Don't recycle wrappers on CUDA; destroy them on Finalize
		}
		outBuffers[i] = &Buffer{
			backend:    e.backend,
			wrapper:    outWrappers[i],
			shape:      actualShape,
			device:     defaultDevice,
			isShared:   true,
			executable: execBackpointer,
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
