// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxruntime

import (
	"os"
	"path/filepath"
	"sync"

	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute-onnx/support/onnxruntime"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
	ort "github.com/gomlx/compute-onnx/internal/ort"
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
	for k, v := range supportedOps {
		ops[k] = v
	}
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

type Backend struct {
	config       string
	cuda         bool
	capabilities compute.Capabilities
	isFinalized  bool
}

var _ compute.Backend = (*Backend)(nil)
var _ compute.DataInterface = (*Backend)(nil)

func init() {
	compute.Register(BackendName, New)
	compute.Register("onnx", New)

	if _, found := os.LookupEnv(NoAutoInstallEnv); found {
		autoInstall = false
	}
}

func New(config string) (compute.Backend, error) {
	configLower := strings.ToLower(config)
	cuda := false
	if configLower == "cuda" || configLower == "gpu" {
		cuda = true
	} else if configLower == "cpu" {
		cuda = false
	} else if configLower == "" {
		// Auto-decide based on GPU hardware presence
		cuda = HasNvidiaGPU()
	} else {
		return nil, errors.Errorf("invalid config value %q: expected \"cpu\", \"cuda\", \"gpu\", or \"\"", config)
	}

	if cuda {
		// Verify CUDA and cuDNN are installed on the system
		if err := checkCUDAAndCUDNN(); err != nil {
			return nil, err
		}
	}

	err := initializeORT(cuda)
	if err != nil {
		return nil, err
	}
	return &Backend{
		config: config,
		cuda:   cuda,
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
		Operations: getSupportedOps(),
		DTypes:     make(map[dtypes.DType]bool),
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
	wrapper, err := newOrtTensorWrapper(shape, flat)
	if err != nil {
		return nil, err
	}
	return &Buffer{
		backend: b,
		wrapper: wrapper,
		shape:   shape,
		device:  deviceNum,
	}, nil
}

func (b *Backend) HasSharedBuffers() bool {
	return false
}

func (b *Backend) NewSharedBuffer(deviceNum compute.DeviceNum, shape shapes.Shape) (compute.Buffer, any, error) {
	return nil, nil, errors.Wrap(compute.ErrNotImplemented, "NewSharedBuffer not implemented")
}

func (b *Backend) Finalize() {
	b.isFinalized = true
}

func (b *Backend) IsFinalized() bool {
	return b.isFinalized
}
