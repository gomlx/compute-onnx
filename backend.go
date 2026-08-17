// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/gomlx/compute"
	ort "github.com/gomlx/compute-onnx/internal/ort"
	"github.com/gomlx/compute-onnx/support/onnxruntime"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

const (
	BackendName = "onnxruntime"
)

var (
	initMutex         sync.Mutex
	isOrtInitialized  bool
	supportedOps      = make(map[compute.OpType]bool)
	supportedOpsMutex sync.RWMutex

	autoInstall = true
)

// NoAutoInstallEnv is the environment variable that, when set, disables
// automatic downloading and installation of prebuilt ONNX Runtime shared libraries.
const NoAutoInstallEnv = "GOMLX_NO_AUTO_INSTALL"

func registerOp(op compute.OpType) {
	supportedOpsMutex.Lock()
	defer supportedOpsMutex.Unlock()
	supportedOps[op] = true
}

func getSupportedOps() map[compute.OpType]bool {
	supportedOpsMutex.RLock()
	defer supportedOpsMutex.RUnlock()
	ops := make(map[compute.OpType]bool, len(supportedOps))
	maps.Copy(ops, supportedOps)
	return ops
}

func initializeORT(cuda bool) error {
	initMutex.Lock()
	defer initMutex.Unlock()
	if isOrtInitialized {
		return nil
	}

	path := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")
	if path == "" {
		installDir, err := onnxruntime.GetInstallPath()
		if err == nil {
			libFilename, err := onnxruntime.GetLibFilename()
			if err == nil {
				targetPath := filepath.Join(installDir, libFilename)
				if _, err := os.Stat(targetPath); err == nil {
					useInstalled := true
					if cuda {
						cudaLibPath := filepath.Join(installDir, "libonnxruntime_providers_cuda.so")
						if _, err := os.Stat(cudaLibPath); err != nil {
							useInstalled = false
						}
					}
					if useInstalled {
						path = targetPath
					}
				}
			}
		}
	}

	if path == "" {
		if !autoInstall {
			return errors.Errorf("ONNX Runtime library not found at ONNXRUNTIME_SHARED_LIBRARY_PATH and auto-installation is disabled via %s", NoAutoInstallEnv)
		}
		var err error
		path, err = onnxruntime.Install(onnxruntime.DefaultVersion, cuda, false)
		if err != nil {
			return errors.Wrap(err, "failed to automatically install ONNX Runtime library")
		}
	}

	ort.SetSharedLibraryPath(path)
	err := ort.InitializeEnvironment()
	if err != nil {
		return errors.Wrap(err, "failed to initialize ONNX Runtime environment")
	}
	isOrtInitialized = true
	return nil
}

// Backend represents a ONNX Runtime backed [compute.Backend].
type Backend struct {
	config         string
	cuda           bool
	logSeverity    int
	capabilities   compute.Capabilities
	isFinalized    bool
	keepModelProto bool
}

// SetKeepModelProto controls whether compiled Executable instances retain the graph *onnx.ModelProto.
// When set to true, executables compiled by this backend store their ONNX ModelProto struct,
// allowing the computation graph to be exported via SaveModel.
func (b *Backend) SetKeepModelProto(keep bool) {
	b.keepModelProto = keep
}

// KeepModelProto returns whether compiled Executable instances retain their graph *onnx.ModelProto.
func (b *Backend) KeepModelProto() bool {
	return b.keepModelProto
}

var _ compute.Backend = (*Backend)(nil)
var _ compute.DataInterface = (*Backend)(nil)

// IsSupportedPlatform returns true if the host OS and CPU architecture are supported by ONNX Runtime binaries.
func IsSupportedPlatform() bool {
	switch runtime.GOOS {
	case "linux", "darwin", "windows":
		switch runtime.GOARCH {
		case "amd64", "arm64":
			return true
		}
	}
	return false
}

func init() {
	if IsSupportedPlatform() {
		compute.Register(BackendName, New)
		compute.Register("onnx", New)
	}

	if _, found := os.LookupEnv(NoAutoInstallEnv); found {
		autoInstall = false
	}
}

func parseConfig(config string) (cuda bool, logSeverity int, err error) {
	cuda = false
	hasProvider := false
	logSeverity = -1 // not set

	if config == "" {
		return HasNvidiaGPU(), -1, nil
	}

	parts := strings.SplitSeq(config, ",")
	for part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if strings.Contains(part, "=") {
			kv := strings.SplitN(part, "=", 2)
			key := strings.ToLower(strings.TrimSpace(kv[0]))
			val := strings.TrimSpace(kv[1])
			if key == "log" {
				var level int
				if _, err := fmt.Sscanf(val, "%d", &level); err != nil {
					return false, 0, errors.Errorf("invalid log level: %q", val)
				}
				// Map level to ORT severity (higher level = more verbose):
				// level=0 -> severity=3 (ERROR)
				// level=1 -> severity=2 (WARNING)
				// level=2 -> severity=1 (INFO)
				// level>=3 -> severity=0 (VERBOSE)
				severity := max(3-level, 0)
				logSeverity = severity
			} else {
				return false, 0, errors.Errorf("unknown config option: %q", key)
			}
		} else {
			partLower := strings.ToLower(part)
			if partLower == "cuda" || partLower == "gpu" {
				cuda = true
				hasProvider = true
			} else if partLower == "cpu" {
				cuda = false
				hasProvider = true
			} else {
				return false, 0, errors.Errorf("invalid config value %q: expected \"cpu\", \"cuda\", \"gpu\", or key=value option", part)
			}
		}
	}

	if !hasProvider {
		cuda = HasNvidiaGPU()
	}
	return cuda, logSeverity, nil
}

func New(config string) (compute.Backend, error) {
	if !IsSupportedPlatform() {
		return nil, errors.Errorf("onnxruntime backend is not supported on platform %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	cuda, logSeverity, err := parseConfig(config)
	if err != nil {
		return nil, err
	}

	if cuda {
		// Verify CUDA and cuDNN are installed on the system
		if err := checkCUDAAndCUDNN(); err != nil {
			return nil, err
		}
	}

	err = initializeORT(cuda)
	if err != nil {
		return nil, err
	}
	return &Backend{
		config:      config,
		cuda:        cuda,
		logSeverity: logSeverity,
	}, nil
}

func (b *Backend) Name() string {
	return BackendName
}

func (b *Backend) String() string {
	return b.Name()
}

func (b *Backend) Description() string {
	if b.cuda {
		return "ONNX Runtime (CUDA GPU) compute backend for GoMLX"
	}
	return "ONNX Runtime (CPU) compute backend for GoMLX"
}

func (b *Backend) NumDevices() int {
	return 1
}

func (b *Backend) DeviceDescription(deviceNum compute.DeviceNum) string {
	if b.cuda {
		return "CUDA GPU (ONNX Runtime Default Device)"
	}
	return "CPU (ONNX Runtime Default Device)"
}

func (b *Backend) Capabilities() compute.Capabilities {
	caps := compute.Capabilities{
		Operations:                  getSupportedOps(),
		DTypes:                      make(map[dtypes.DType]bool),
		PreferConstantsForVariables: true,
		DynamicAxes:                 true,
	}
	caps.DTypes[dtypes.Float32] = true
	caps.DTypes[dtypes.Float64] = true
	caps.DTypes[dtypes.Float16] = true
	caps.DTypes[dtypes.BFloat16] = true
	caps.DTypes[dtypes.Int32] = true
	caps.DTypes[dtypes.Int64] = true
	caps.DTypes[dtypes.Bool] = true
	caps.DTypes[dtypes.Int8] = true
	caps.DTypes[dtypes.Uint8] = true
	caps.DTypes[dtypes.Int16] = true
	caps.DTypes[dtypes.Uint16] = true
	caps.DTypes[dtypes.Uint32] = true
	caps.DTypes[dtypes.Uint64] = true
	return caps
}

func (b *Backend) Builder(name string) compute.Builder {
	return NewBuilder(name, b)
}

func (b *Backend) BufferFromFlatData(deviceNum compute.DeviceNum, flat any, shape shapes.Shape) (compute.Buffer, error) {
	return b.bufferFromFlatDataCPU(deviceNum, flat, shape)
}

func (b *Backend) bufferFromFlatDataCPU(deviceNum compute.DeviceNum, flat any, shape shapes.Shape) (*Buffer, error) {
	wrapper, err := newOrtTensorWrapper(shape, flat)
	if err != nil {
		return nil, err
	}
	return &Buffer{
		backend:  b,
		wrapper:  wrapper,
		shape:    shape,
		device:   deviceNum,
		isShared: true,
	}, nil
}

func (b *Backend) HasSharedBuffers() bool {
	return !b.cuda
}

func (b *Backend) NewSharedBuffer(deviceNum compute.DeviceNum, shape shapes.Shape) (compute.Buffer, any, error) {
	ortShape := ort.NewShape(toInt64s(shape.Dimensions)...)
	var wrapper ortTensorWrapper
	var flat any

	switch shape.DType {
	case dtypes.Float32:
		t, err := ort.NewEmptyTensor[float32](ortShape)
		if err != nil {
			return nil, nil, err
		}
		wrapper = &typedTensor[float32]{tensor: t}
		flat = t.GetData()
	case dtypes.Float64:
		t, err := ort.NewEmptyTensor[float64](ortShape)
		if err != nil {
			return nil, nil, err
		}
		wrapper = &typedTensor[float64]{tensor: t}
		flat = t.GetData()
	case dtypes.Int32:
		t, err := ort.NewEmptyTensor[int32](ortShape)
		if err != nil {
			return nil, nil, err
		}
		wrapper = &typedTensor[int32]{tensor: t}
		flat = t.GetData()
	case dtypes.Int64:
		t, err := ort.NewEmptyTensor[int64](ortShape)
		if err != nil {
			return nil, nil, err
		}
		wrapper = &typedTensor[int64]{tensor: t}
		flat = t.GetData()
	case dtypes.Bool:
		t, err := ort.NewEmptyTensor[bool](ortShape)
		if err != nil {
			return nil, nil, err
		}
		wrapper = &typedTensor[bool]{tensor: t}
		flat = t.GetData()
	case dtypes.Int8:
		t, err := ort.NewEmptyTensor[int8](ortShape)
		if err != nil {
			return nil, nil, err
		}
		wrapper = &typedTensor[int8]{tensor: t}
		flat = t.GetData()
	case dtypes.Uint8:
		t, err := ort.NewEmptyTensor[uint8](ortShape)
		if err != nil {
			return nil, nil, err
		}
		wrapper = &typedTensor[uint8]{tensor: t}
		flat = t.GetData()
	case dtypes.Int16:
		t, err := ort.NewEmptyTensor[int16](ortShape)
		if err != nil {
			return nil, nil, err
		}
		wrapper = &typedTensor[int16]{tensor: t}
		flat = t.GetData()
	case dtypes.Uint16:
		t, err := ort.NewEmptyTensor[uint16](ortShape)
		if err != nil {
			return nil, nil, err
		}
		wrapper = &typedTensor[uint16]{tensor: t}
		flat = t.GetData()
	case dtypes.Uint32:
		t, err := ort.NewEmptyTensor[uint32](ortShape)
		if err != nil {
			return nil, nil, err
		}
		wrapper = &typedTensor[uint32]{tensor: t}
		flat = t.GetData()
	case dtypes.Uint64:
		t, err := ort.NewEmptyTensor[uint64](ortShape)
		if err != nil {
			return nil, nil, err
		}
		wrapper = &typedTensor[uint64]{tensor: t}
		flat = t.GetData()
	default:
		return nil, nil, errors.Errorf("shared buffers not implemented for dtype %s", shape.DType)
	}

	buf := &Buffer{
		backend:  b,
		wrapper:  wrapper,
		shape:    shape,
		device:   deviceNum,
		isShared: true,
	}
	return buf, flat, nil
}

func (b *Backend) Finalize() {
	b.isFinalized = true
}

func (b *Backend) IsFinalized() bool {
	return b.isFinalized
}
