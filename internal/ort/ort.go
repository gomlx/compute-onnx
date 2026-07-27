// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ort

/*
#include "onnxruntime_c_api.h"

// Forward declarations of our wrappers:
const OrtApi* GetOrtApi(void* get_api_base_ptr, uint32_t version);

OrtStatus* wrapper_CreateEnv(const OrtApi* api, OrtLoggingLevel default_logging_level, const char* logid, OrtEnv** out);
void wrapper_ReleaseEnv(const OrtApi* api, OrtEnv* env);

OrtStatus* wrapper_CreateSessionOptions(const OrtApi* api, OrtSessionOptions** out);
void wrapper_ReleaseSessionOptions(const OrtApi* api, OrtSessionOptions* options);

OrtStatus* wrapper_CreateSessionWithONNXData(const OrtApi* api, const OrtEnv* env, const void* model_data, size_t model_data_length, const OrtSessionOptions* options, OrtSession** out);
void wrapper_ReleaseSession(const OrtApi* api, OrtSession* session);

OrtStatus* wrapper_SessionGetInputCount(const OrtApi* api, const OrtSession* session, size_t* out);
OrtStatus* wrapper_SessionGetOutputCount(const OrtApi* api, const OrtSession* session, size_t* out);

OrtStatus* wrapper_SessionGetInputName(const OrtApi* api, const OrtSession* session, size_t index, OrtAllocator* allocator, char** value);
OrtStatus* wrapper_SessionGetOutputName(const OrtApi* api, const OrtSession* session, size_t index, OrtAllocator* allocator, char** value);

OrtStatus* wrapper_CreateCpuMemoryInfo(const OrtApi* api, OrtAllocatorType type, OrtMemType mem_type, OrtMemoryInfo** out);
void wrapper_ReleaseMemoryInfo(const OrtApi* api, OrtMemoryInfo* info);

OrtStatus* wrapper_CreateTensorWithDataAsOrtValue(const OrtApi* api, const OrtMemoryInfo* info, void* p_data, size_t p_data_len, const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type, OrtValue** out);
OrtStatus* wrapper_CreateTensorAsOrtValue(const OrtApi* api, OrtAllocator* allocator, const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type, OrtValue** out);
OrtStatus* wrapper_GetTensorMutableData(const OrtApi* api, OrtValue* value, void** out);
void wrapper_ReleaseValue(const OrtApi* api, OrtValue* value);

OrtStatus* wrapper_Run(const OrtApi* api, OrtSession* session, const OrtRunOptions* run_options, const char* const* input_names, const OrtValue* const* input_values, size_t input_count, const char* const* output_names, size_t output_count, OrtValue** output_values);

const char* wrapper_GetErrorMessage(const OrtApi* api, const OrtStatus* status);
void wrapper_ReleaseStatus(const OrtApi* api, OrtStatus* status);

OrtStatus* wrapper_GetAllocatorWithDefaultOptions(const OrtApi* api, OrtAllocator** out);

OrtStatus* wrapper_CreateCUDAProviderOptions(const OrtApi* api, OrtCUDAProviderOptionsV2** out);
OrtStatus* wrapper_SessionOptionsAppendExecutionProvider_CUDA_V2(const OrtApi* api, OrtSessionOptions* options, const OrtCUDAProviderOptionsV2* cuda_options);
void wrapper_ReleaseCUDAProviderOptions(const OrtApi* api, OrtCUDAProviderOptionsV2* input);

OrtStatus* wrapper_GetTensorTypeAndShape(const OrtApi* api, const OrtValue* value, OrtTensorTypeAndShapeInfo** out);
OrtStatus* wrapper_GetTensorElementType(const OrtApi* api, const OrtTensorTypeAndShapeInfo* info, ONNXTensorElementDataType* out);
OrtStatus* wrapper_GetDimensionsCount(const OrtApi* api, const OrtTensorTypeAndShapeInfo* info, size_t* out);
OrtStatus* wrapper_GetDimensions(const OrtApi* api, const OrtTensorTypeAndShapeInfo* info, int64_t* dim_values, size_t dim_values_length);
void wrapper_ReleaseTensorTypeAndShapeInfo(const OrtApi* api, OrtTensorTypeAndShapeInfo* info);
OrtStatus* wrapper_SetSessionLogSeverityLevel(const OrtApi* api, OrtSessionOptions* options, int session_log_severity_level);

// IoBinding
OrtStatus* wrapper_CreateIoBinding(const OrtApi* api, OrtSession* session, OrtIoBinding** out);
void wrapper_ReleaseIoBinding(const OrtApi* api, OrtIoBinding* binding);
OrtStatus* wrapper_BindInput(const OrtApi* api, OrtIoBinding* binding, const char* name, const OrtValue* val);
OrtStatus* wrapper_BindOutput(const OrtApi* api, OrtIoBinding* binding, const char* name, const OrtValue* val);
OrtStatus* wrapper_BindOutputToDevice(const OrtApi* api, OrtIoBinding* binding, const char* name, const OrtMemoryInfo* mem_info);
OrtStatus* wrapper_RunWithBinding(const OrtApi* api, OrtSession* session, const OrtRunOptions* run_options, const OrtIoBinding* binding);
OrtStatus* wrapper_GetBoundOutputValues(const OrtApi* api, const OrtIoBinding* binding, OrtAllocator* allocator, OrtValue*** output, size_t* output_count);
void wrapper_ClearBoundInputs(const OrtApi* api, OrtIoBinding* binding);
void wrapper_ClearBoundOutputs(const OrtApi* api, OrtIoBinding* binding);
OrtStatus* wrapper_SynchronizeBoundInputs(const OrtApi* api, OrtIoBinding* binding);
OrtStatus* wrapper_SynchronizeBoundOutputs(const OrtApi* api, OrtIoBinding* binding);

// MemoryInfo + Allocator
OrtStatus* wrapper_CreateMemoryInfo(const OrtApi* api, const char* name, enum OrtAllocatorType type, int id, enum OrtMemType mem_type, OrtMemoryInfo** out);
OrtStatus* wrapper_CreateAllocator(const OrtApi* api, const OrtSession* session, const OrtMemoryInfo* mem_info, OrtAllocator** out);
void wrapper_ReleaseAllocator(const OrtApi* api, OrtAllocator* allocator);
OrtStatus* wrapper_AllocatorAlloc(const OrtApi* api, OrtAllocator* allocator, size_t size, void** out);
OrtStatus* wrapper_AllocatorFree(const OrtApi* api, OrtAllocator* allocator, void* p);
*/
import "C"
import (
	"fmt"
	"unsafe"
)

var (
	ortLibPath string
	ortApi     *C.OrtApi
	ortHandle  unsafe.Pointer
	ortEnv     *Env
)

func SetSharedLibraryPath(path string) {
	ortLibPath = path
}

func InitializeEnvironment() error {
	if ortApi != nil {
		return nil
	}
	if ortLibPath == "" {
		return fmt.Errorf("shared library path not set")
	}

	handle, err := loadLibrary(ortLibPath)
	if err != nil {
		return err
	}
	ortHandle = handle

	apiBasePtr, err := getOrtGetApiBase(handle)
	if err != nil {
		_ = closeLibrary(handle)
		return err
	}

	ortApi = (*C.OrtApi)(C.GetOrtApi(apiBasePtr, C.uint32_t(26))) // version 26
	if ortApi == nil {
		_ = closeLibrary(handle)
		return fmt.Errorf("failed to get OrtApi version 26")
	}

	// Create default global environment
	env, err := NewEnv("gomlx_ort_env")
	if err != nil {
		_ = closeLibrary(handle)
		ortApi = nil
		return err
	}
	ortEnv = env

	return nil
}

func statusToError(status *C.OrtStatus) error {
	if status == nil {
		return nil
	}
	defer C.wrapper_ReleaseStatus(ortApi, status)
	msgC := C.wrapper_GetErrorMessage(ortApi, status)
	return fmt.Errorf("ONNX Runtime error: %s", C.GoString(msgC))
}

type Env struct {
	env *C.OrtEnv
}

func NewEnv(logID string) (*Env, error) {
	logIDC := C.CString(logID)
	defer C.free(unsafe.Pointer(logIDC))

	var env *C.OrtEnv
	status := C.wrapper_CreateEnv(ortApi, C.ORT_LOGGING_LEVEL_WARNING, logIDC, &env)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return &Env{env: env}, nil
}

func (e *Env) Destroy() error {
	if e.env != nil {
		C.wrapper_ReleaseEnv(ortApi, e.env)
		e.env = nil
	}
	return nil
}

type SessionOptions struct {
	options *C.OrtSessionOptions
}

func NewSessionOptions() (*SessionOptions, error) {
	var options *C.OrtSessionOptions
	status := C.wrapper_CreateSessionOptions(ortApi, &options)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return &SessionOptions{options: options}, nil
}

func (so *SessionOptions) Destroy() error {
	if so.options != nil {
		C.wrapper_ReleaseSessionOptions(ortApi, so.options)
		so.options = nil
	}
	return nil
}

func (so *SessionOptions) AppendExecutionProviderCUDA(cudaOpts *CUDAProviderOptions) error {
	status := C.wrapper_SessionOptionsAppendExecutionProvider_CUDA_V2(ortApi, so.options, cudaOpts.cudaOpts)
	return statusToError(status)
}

func (so *SessionOptions) SetSessionLogSeverityLevel(level int) error {
	status := C.wrapper_SetSessionLogSeverityLevel(ortApi, so.options, C.int(level))
	return statusToError(status)
}

type CUDAProviderOptions struct {
	cudaOpts *C.OrtCUDAProviderOptionsV2
}

func NewCUDAProviderOptions() (*CUDAProviderOptions, error) {
	var cudaOpts *C.OrtCUDAProviderOptionsV2
	status := C.wrapper_CreateCUDAProviderOptions(ortApi, &cudaOpts)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return &CUDAProviderOptions{cudaOpts: cudaOpts}, nil
}

func (c *CUDAProviderOptions) Destroy() error {
	if c.cudaOpts != nil {
		C.wrapper_ReleaseCUDAProviderOptions(ortApi, c.cudaOpts)
		c.cudaOpts = nil
	}
	return nil
}

type Session struct {
	session *C.OrtSession
}

func NewSessionWithONNXData(env *Env, modelBytes []byte, options *SessionOptions) (*Session, error) {
	var session *C.OrtSession
	var optPtr *C.OrtSessionOptions
	if options != nil {
		optPtr = options.options
	}
	status := C.wrapper_CreateSessionWithONNXData(ortApi, env.env, unsafe.Pointer(&modelBytes[0]), C.size_t(len(modelBytes)), optPtr, &session)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return &Session{session: session}, nil
}

func (s *Session) Destroy() error {
	if s.session != nil {
		C.wrapper_ReleaseSession(ortApi, s.session)
		s.session = nil
	}
	return nil
}

func (s *Session) GetInputCount() (int, error) {
	var count C.size_t
	status := C.wrapper_SessionGetInputCount(ortApi, s.session, &count)
	if err := statusToError(status); err != nil {
		return 0, err
	}
	return int(count), nil
}

func (s *Session) GetOutputCount() (int, error) {
	var count C.size_t
	status := C.wrapper_SessionGetOutputCount(ortApi, s.session, &count)
	if err := statusToError(status); err != nil {
		return 0, err
	}
	return int(count), nil
}

func (s *Session) GetInputName(index int, allocator *Allocator) (string, error) {
	var name *C.char
	status := C.wrapper_SessionGetInputName(ortApi, s.session, C.size_t(index), allocator.allocator, &name)
	if err := statusToError(status); err != nil {
		return "", err
	}
	defer C.free(unsafe.Pointer(name))
	return C.GoString(name), nil
}

func (s *Session) GetOutputName(index int, allocator *Allocator) (string, error) {
	var name *C.char
	status := C.wrapper_SessionGetOutputName(ortApi, s.session, C.size_t(index), allocator.allocator, &name)
	if err := statusToError(status); err != nil {
		return "", err
	}
	defer C.free(unsafe.Pointer(name))
	return C.GoString(name), nil
}

type Allocator struct {
	allocator *C.OrtAllocator
}

func NewDefaultAllocator() (*Allocator, error) {
	var allocator *C.OrtAllocator
	status := C.wrapper_GetAllocatorWithDefaultOptions(ortApi, &allocator)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return &Allocator{allocator: allocator}, nil
}

type MemoryInfo struct {
	info *C.OrtMemoryInfo
}

func NewCpuMemoryInfo() (*MemoryInfo, error) {
	var info *C.OrtMemoryInfo
	status := C.wrapper_CreateCpuMemoryInfo(ortApi, C.OrtDeviceAllocator, C.OrtMemTypeDefault, &info)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return &MemoryInfo{info: info}, nil
}

func (mi *MemoryInfo) Destroy() error {
	if mi.info != nil {
		C.wrapper_ReleaseMemoryInfo(ortApi, mi.info)
		mi.info = nil
	}
	return nil
}

type DataType int

const (
	TensorElementDataTypeFloat    DataType = 1
	TensorElementDataTypeUint8    DataType = 2
	TensorElementDataTypeInt8     DataType = 3
	TensorElementDataTypeUint16   DataType = 4
	TensorElementDataTypeInt16    DataType = 5
	TensorElementDataTypeInt32    DataType = 6
	TensorElementDataTypeInt64    DataType = 7
	TensorElementDataTypeBool     DataType = 9
	TensorElementDataTypeFloat16  DataType = 10
	TensorElementDataTypeDouble   DataType = 11
	TensorElementDataTypeUint32   DataType = 12
	TensorElementDataTypeUint64   DataType = 13
	TensorElementDataTypeBFloat16 DataType = 16
)

type Shape []int64

func NewShape(dims ...int64) Shape {
	return Shape(dims)
}

type TensorData interface {
	~float32 | ~float64 | ~int32 | ~int64 | ~bool | ~int8 | ~uint8 | ~int16 | ~uint16 | ~uint32 | ~uint64
}

type Value interface {
	GetTensorMutableData() (unsafe.Pointer, error)
	Destroy() error
	cValue() *C.OrtValue
}

// RawValue wraps a raw *C.OrtValue, typically a GPU-resident tensor from IoBinding.
type RawValue struct {
	val *C.OrtValue
}

// WrapRawOrtValue creates a RawValue from a raw *C.OrtValue pointer.
// The returned Value takes ownership and will release the OrtValue on Destroy().
func WrapRawOrtValue(v *C.OrtValue) Value {
	return &RawValue{val: v}
}

func (r *RawValue) GetTensorMutableData() (unsafe.Pointer, error) {
	var out unsafe.Pointer
	status := C.wrapper_GetTensorMutableData(ortApi, r.val, &out)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return out, nil
}

func (r *RawValue) Destroy() error {
	if r.val != nil {
		C.wrapper_ReleaseValue(ortApi, r.val)
		r.val = nil
	}
	return nil
}

func (r *RawValue) cValue() *C.OrtValue {
	return r.val
}

type Tensor[T TensorData] struct {
	val   *C.OrtValue
	shape Shape
}

func (t *Tensor[T]) GetShape() Shape {
	return t.shape
}

func (t *Tensor[T]) cValue() *C.OrtValue {
	return t.val
}

func (t *Tensor[T]) GetTensorMutableData() (unsafe.Pointer, error) {
	var out unsafe.Pointer
	status := C.wrapper_GetTensorMutableData(ortApi, t.val, &out)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return out, nil
}

func (t *Tensor[T]) GetData() []T {
	dataPtr, err := t.GetTensorMutableData()
	if err != nil {
		panic(err)
	}
	size := 1
	for _, dim := range t.shape {
		size *= int(dim)
	}
	if size == 0 {
		return []T{}
	}
	if dataPtr == nil {
		panic(fmt.Sprintf("GetData: dataPtr is nil but size is %d, shape is %v", size, t.shape))
	}
	return unsafe.Slice((*T)(dataPtr), size)
}

func (t *Tensor[T]) updateVal(newVal *C.OrtValue) {
	t.val = newVal
}

func (t *Tensor[T]) Destroy() error {
	if t.val != nil {
		C.wrapper_ReleaseValue(ortApi, t.val)
		t.val = nil
	}
	return nil
}

func NewTensor[T TensorData](shape Shape, data []T) (*Tensor[T], error) {
	memInfo, err := NewCpuMemoryInfo()
	if err != nil {
		return nil, err
	}
	defer memInfo.Destroy()

	var val *C.OrtValue
	var shapePtr *C.int64_t
	if len(shape) > 0 {
		shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}

	var dtype DataType
	var dummy T
	switch any(dummy).(type) {
	case float32:
		dtype = TensorElementDataTypeFloat
	case float64:
		dtype = TensorElementDataTypeDouble
	case int32:
		dtype = TensorElementDataTypeInt32
	case int64:
		dtype = TensorElementDataTypeInt64
	case bool:
		dtype = TensorElementDataTypeBool
	case int8:
		dtype = TensorElementDataTypeInt8
	case uint8:
		dtype = TensorElementDataTypeUint8
	case int16:
		dtype = TensorElementDataTypeInt16
	case uint16:
		dtype = TensorElementDataTypeUint16
	case uint32:
		dtype = TensorElementDataTypeUint32
	case uint64:
		dtype = TensorElementDataTypeUint64
	default:
		return nil, fmt.Errorf("unsupported tensor type %T", dummy)
	}

	dataLen := len(data) * int(unsafe.Sizeof(dummy))
	var dataPtr unsafe.Pointer
	if len(data) > 0 {
		dataPtr = unsafe.Pointer(&data[0])
	}
	status := C.wrapper_CreateTensorWithDataAsOrtValue(
		ortApi,
		memInfo.info,
		dataPtr,
		C.size_t(dataLen),
		shapePtr,
		C.size_t(len(shape)),
		C.ONNXTensorElementDataType(dtype),
		&val,
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}

	return &Tensor[T]{
		val:   val,
		shape: shape,
	}, nil
}

func NewEmptyTensor[T TensorData](shape Shape) (*Tensor[T], error) {
	allocator, err := NewDefaultAllocator()
	if err != nil {
		return nil, err
	}

	var val *C.OrtValue
	var shapePtr *C.int64_t
	if len(shape) > 0 {
		shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}

	var dtype DataType
	var dummy T
	switch any(dummy).(type) {
	case float32:
		dtype = TensorElementDataTypeFloat
	case float64:
		dtype = TensorElementDataTypeDouble
	case int32:
		dtype = TensorElementDataTypeInt32
	case int64:
		dtype = TensorElementDataTypeInt64
	case bool:
		dtype = TensorElementDataTypeBool
	case int8:
		dtype = TensorElementDataTypeInt8
	case uint8:
		dtype = TensorElementDataTypeUint8
	case int16:
		dtype = TensorElementDataTypeInt16
	case uint16:
		dtype = TensorElementDataTypeUint16
	case uint32:
		dtype = TensorElementDataTypeUint32
	case uint64:
		dtype = TensorElementDataTypeUint64
	default:
		return nil, fmt.Errorf("unsupported tensor type %T", dummy)
	}

	status := C.wrapper_CreateTensorAsOrtValue(
		ortApi,
		allocator.allocator,
		shapePtr,
		C.size_t(len(shape)),
		C.ONNXTensorElementDataType(dtype),
		&val,
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}

	return &Tensor[T]{
		val:   val,
		shape: shape,
	}, nil
}

func (s *Session) Run(inputNames []string, inputValues []Value, outputNames []string, outputValues []Value) error {
	arena := GetArena(8192)
	defer ReturnArena(arena)

	nInputs := len(inputNames)
	nOutputs := len(outputNames)

	inputNamesPtr := arena.AllocCharStarSlice(nInputs)
	inputValuesPtr := arena.AllocOrtValueStarSlice(nInputs)
	outputNamesPtr := arena.AllocCharStarSlice(nOutputs)
	outputValuesPtr := arena.AllocOrtValueStarSlice(nOutputs)

	inputNamesSlice := unsafe.Slice(inputNamesPtr, nInputs)
	inputValuesSlice := unsafe.Slice(inputValuesPtr, nInputs)
	outputNamesSlice := unsafe.Slice(outputNamesPtr, nOutputs)
	outputValuesSlice := unsafe.Slice(outputValuesPtr, nOutputs)

	for i := 0; i < nInputs; i++ {
		inputNamesSlice[i] = arena.AllocCString(inputNames[i])
		inputValuesSlice[i] = inputValues[i].cValue()
	}

	for i := 0; i < nOutputs; i++ {
		outputNamesSlice[i] = arena.AllocCString(outputNames[i])
		if outputValues[i] != nil {
			outputValuesSlice[i] = outputValues[i].cValue()
		} else {
			outputValuesSlice[i] = nil
		}
	}

	status := C.wrapper_Run(
		ortApi,
		s.session,
		nil, // OrtRunOptions
		inputNamesPtr,
		inputValuesPtr,
		C.size_t(nInputs),
		outputNamesPtr,
		C.size_t(nOutputs),
		outputValuesPtr,
	)

	if err := statusToError(status); err != nil {
		return err
	}

	// Wrap any automatically-allocated outputs or update overridden pre-allocated outputs
	for i := 0; i < nOutputs; i++ {
		if outputValues[i] != nil {
			if t, ok := outputValues[i].(interface{ updateVal(*C.OrtValue) }); ok {
				t.updateVal(outputValuesSlice[i])
			}
			continue
		}
		val, err := createGoValueFromOrtValue(outputValuesSlice[i])
		if err != nil {
			return err
		}
		outputValues[i] = val
	}
	return nil
}

type DynamicAdvancedSession struct {
	session     *Session
	inputNames  []string
	outputNames []string

	// Cached C-strings for input/output names, allocated once at session creation.
	cInputNames  []*C.char
	cOutputNames []*C.char
}

func NewDynamicAdvancedSessionWithONNXData(modelBytes []byte, inputNames []string, outputNames []string, options *SessionOptions) (*DynamicAdvancedSession, error) {
	session, err := NewSessionWithONNXData(ortEnv, modelBytes, options)
	if err != nil {
		return nil, err
	}

	// Pre-allocate persistent C strings for input/output names.
	cInputNames := make([]*C.char, len(inputNames))
	for i, name := range inputNames {
		cInputNames[i] = C.CString(name)
	}
	cOutputNames := make([]*C.char, len(outputNames))
	for i, name := range outputNames {
		cOutputNames[i] = C.CString(name)
	}

	return &DynamicAdvancedSession{
		session:      session,
		inputNames:   inputNames,
		outputNames:  outputNames,
		cInputNames:  cInputNames,
		cOutputNames: cOutputNames,
	}, nil
}

// CreateIoBinding creates an IoBinding for this session.
func (s *DynamicAdvancedSession) CreateIoBinding() (*IoBinding, error) {
	return NewIoBinding(s.session)
}

// CInputNames returns the pre-allocated C-strings for input names.
func (s *DynamicAdvancedSession) CInputNames() []*C.char {
	return s.cInputNames
}

// COutputNames returns the pre-allocated C-strings for output names.
func (s *DynamicAdvancedSession) COutputNames() []*C.char {
	return s.cOutputNames
}

func (s *DynamicAdvancedSession) Destroy() error {
	for _, cs := range s.cInputNames {
		C.free(unsafe.Pointer(cs))
	}
	s.cInputNames = nil
	for _, cs := range s.cOutputNames {
		C.free(unsafe.Pointer(cs))
	}
	s.cOutputNames = nil
	s.session.Destroy()
	return nil
}

func (s *DynamicAdvancedSession) Run(inputs []Value, outputs []Value) error {
	nInputs := len(s.inputNames)
	nOutputs := len(s.outputNames)

	// Arena needs space for 4 pointer arrays: 2*(nInputs+nOutputs) pointers, each 8 bytes on 64-bit,
	// plus alignment padding.
	arenaSize := (nInputs + nOutputs) * 16
	if arenaSize < 2048 {
		arenaSize = 2048
	}
	arena := GetArena(arenaSize)
	defer ReturnArena(arena)

	inputNamesPtr := arena.AllocCharStarSlice(nInputs)
	inputValuesPtr := arena.AllocOrtValueStarSlice(nInputs)
	outputNamesPtr := arena.AllocCharStarSlice(nOutputs)
	outputValuesPtr := arena.AllocOrtValueStarSlice(nOutputs)

	inputNamesSlice := unsafe.Slice(inputNamesPtr, nInputs)
	inputValuesSlice := unsafe.Slice(inputValuesPtr, nInputs)
	outputNamesSlice := unsafe.Slice(outputNamesPtr, nOutputs)
	outputValuesSlice := unsafe.Slice(outputValuesPtr, nOutputs)

	for i := 0; i < nInputs; i++ {
		inputNamesSlice[i] = s.cInputNames[i]
		inputValuesSlice[i] = inputs[i].cValue()
	}

	for i := 0; i < nOutputs; i++ {
		outputNamesSlice[i] = s.cOutputNames[i]
		if outputs[i] != nil {
			outputValuesSlice[i] = outputs[i].cValue()
		} else {
			outputValuesSlice[i] = nil
		}
	}

	status := C.wrapper_Run(
		ortApi,
		s.session.session,
		nil, // OrtRunOptions
		inputNamesPtr,
		inputValuesPtr,
		C.size_t(nInputs),
		outputNamesPtr,
		C.size_t(nOutputs),
		outputValuesPtr,
	)

	if err := statusToError(status); err != nil {
		return err
	}

	// Wrap any automatically-allocated outputs or update overridden pre-allocated outputs
	for i := 0; i < nOutputs; i++ {
		if outputs[i] != nil {
			if t, ok := outputs[i].(interface{ updateVal(*C.OrtValue) }); ok {
				t.updateVal(outputValuesSlice[i])
			}
			continue
		}
		val, err := createGoValueFromOrtValue(outputValuesSlice[i])
		if err != nil {
			return err
		}
		outputs[i] = val
	}
	return nil
}

func createGoValueFromOrtValue(v *C.OrtValue) (Value, error) {
	if v == nil {
		return nil, fmt.Errorf("createGoValueFromOrtValue received nil pointer")
	}

	var info *C.OrtTensorTypeAndShapeInfo
	status := C.wrapper_GetTensorTypeAndShape(ortApi, v, &info)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	defer C.wrapper_ReleaseTensorTypeAndShapeInfo(ortApi, info)

	var elemType C.ONNXTensorElementDataType
	status = C.wrapper_GetTensorElementType(ortApi, info, &elemType)
	if err := statusToError(status); err != nil {
		return nil, err
	}

	var dimCount C.size_t
	status = C.wrapper_GetDimensionsCount(ortApi, info, &dimCount)
	if err := statusToError(status); err != nil {
		return nil, err
	}

	shape := make([]int64, dimCount)
	if dimCount > 0 {
		status = C.wrapper_GetDimensions(ortApi, info, (*C.int64_t)(unsafe.Pointer(&shape[0])), dimCount)
		if err := statusToError(status); err != nil {
			return nil, err
		}
	}

	switch DataType(elemType) {
	case TensorElementDataTypeFloat:
		return &Tensor[float32]{val: v, shape: shape}, nil
	case TensorElementDataTypeDouble:
		return &Tensor[float64]{val: v, shape: shape}, nil
	case TensorElementDataTypeInt32:
		return &Tensor[int32]{val: v, shape: shape}, nil
	case TensorElementDataTypeInt64:
		return &Tensor[int64]{val: v, shape: shape}, nil
	case TensorElementDataTypeBool:
		return &Tensor[bool]{val: v, shape: shape}, nil
	case TensorElementDataTypeInt8:
		return &Tensor[int8]{val: v, shape: shape}, nil
	case TensorElementDataTypeUint8:
		return &Tensor[uint8]{val: v, shape: shape}, nil
	case TensorElementDataTypeInt16:
		return &Tensor[int16]{val: v, shape: shape}, nil
	case TensorElementDataTypeUint16:
		return &Tensor[uint16]{val: v, shape: shape}, nil
	case TensorElementDataTypeUint32:
		return &Tensor[uint32]{val: v, shape: shape}, nil
	case TensorElementDataTypeUint64:
		return &Tensor[uint64]{val: v, shape: shape}, nil
	default:
		return nil, fmt.Errorf("unsupported tensor element type: %d", elemType)
	}
}
