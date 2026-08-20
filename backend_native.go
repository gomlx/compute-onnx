// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !js

package onnxbackend

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute-onnx/internal/device/cuda"
	"github.com/gomlx/compute-onnx/internal/engine/native"
	ort "github.com/gomlx/compute-onnx/internal/ort"
	"github.com/gomlx/compute-onnx/support/onnxruntime"
	onnx "github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// Type aliases for native engine types.
type Executable = native.Executable
type Buffer = native.Buffer

const SaveOnFailureEnv = native.SaveOnFailureEnv

var (
	initMutex        sync.Mutex
	isOrtInitialized bool
	autoInstall      = true
)

// NoAutoInstallEnv is the environment variable that, when set, disables
// automatic downloading and installation of prebuilt ONNX Runtime shared libraries.
const NoAutoInstallEnv = "GOMLX_NO_AUTO_INSTALL"

// EnableAutoInstall controls whether the backend automatically downloads and installs
// prebuilt ONNX Runtime shared libraries if none are found.
func EnableAutoInstall(enable bool) {
	initMutex.Lock()
	defer initMutex.Unlock()
	autoInstall = enable
}

func init() {
	if _, found := os.LookupEnv(NoAutoInstallEnv); found {
		autoInstall = false
	}
}

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

func isLibraryPath(part string) bool {
	partLower := strings.ToLower(part)
	if strings.HasSuffix(partLower, ".so") || strings.HasSuffix(partLower, ".dylib") || strings.HasSuffix(partLower, ".dll") {
		return true
	}
	if strings.Contains(part, "/") || strings.Contains(part, "\\") {
		return true
	}
	if info, err := os.Stat(part); err == nil && !info.IsDir() {
		return true
	}
	return false
}

func initializeORT(cudaEnabled bool, customLibPath string) error {
	initMutex.Lock()
	defer initMutex.Unlock()
	if isOrtInitialized {
		return nil
	}

	path := customLibPath
	if path == "" {
		path = os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")
		if path == "" {
			installDir, err := onnxruntime.GetInstallPath()
			if err == nil {
				libFilename, err := onnxruntime.GetLibFilename()
				if err == nil {
					targetPath := filepath.Join(installDir, libFilename)
					if _, err := os.Stat(targetPath); err == nil {
						useInstalled := true
						if cudaEnabled {
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
	}

	if path != "" {
		if _, err := os.Stat(path); err != nil {
			return errors.Wrapf(err, "ONNX Runtime library not found at specified path %q", path)
		}
	} else {
		if !autoInstall {
			return errors.Errorf("ONNX Runtime library not found (ONNXRUNTIME_SHARED_LIBRARY_PATH is not set) and auto-installation is disabled via %s or EnableAutoInstall(false)", NoAutoInstallEnv)
		}
		var err error
		path, err = onnxruntime.Install(onnxruntime.DefaultVersion, cudaEnabled, "", "", false)
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

func parseConfig(config string) (cudaEnabled bool, logSeverity int, customLibPath string, err error) {
	cudaEnabled = false
	hasProvider := false
	logSeverity = -1 // not set

	if config == "" {
		if envVal := os.Getenv("GOMLX_BACKEND"); envVal != "" {
			var errEnv error
			config, errEnv = ParseGOMLXBackendEnv(envVal)
			if errEnv != nil {
				return false, 0, "", errEnv
			}
		}
	} else if strings.Contains(config, ":") || strings.EqualFold(config, "onnx") || strings.EqualFold(config, "onnxruntime") {
		parsed, errEnv := ParseGOMLXBackendEnv(config)
		if errEnv == nil {
			config = parsed
		} else if !isLibraryPath(config) && !strings.Contains(config, "=") {
			return false, 0, "", errEnv
		}
	}

	config = strings.TrimSpace(config)
	if config == "" {
		if envPath := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); envPath != "" {
			return cuda.HasNvidiaGPU() && cuda.IsCUDALibraryAvailable(filepath.Dir(envPath)), -1, "", nil
		}
		return cuda.HasNvidiaGPU(), -1, "", nil
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
					return false, 0, "", errors.Errorf("invalid log level: %q", val)
				}
				severity := max(3-level, 0)
				logSeverity = severity
			} else {
				return false, 0, "", errors.Errorf("unknown config option: %q", key)
			}
		} else {
			partLower := strings.ToLower(part)
			if partLower == "cuda" || partLower == "gpu" {
				cudaEnabled = true
				hasProvider = true
			} else if partLower == "cpu" {
				cudaEnabled = false
				hasProvider = true
			} else if isLibraryPath(part) {
				customLibPath = part
			} else {
				return false, 0, "", errors.Errorf("invalid config value %q: expected \"cpu\", \"cuda\", \"gpu\", path to ORT library, or key=value option", part)
			}
		}
	}

	if !hasProvider {
		if customLibPath != "" {
			cudaEnabled = cuda.HasNvidiaGPU() && cuda.IsCUDALibraryAvailable(filepath.Dir(customLibPath))
		} else if envPath := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); envPath != "" {
			cudaEnabled = cuda.HasNvidiaGPU() && cuda.IsCUDALibraryAvailable(filepath.Dir(envPath))
		} else {
			cudaEnabled = cuda.HasNvidiaGPU()
		}
	}
	return cudaEnabled, logSeverity, customLibPath, nil
}

// New creates a new ONNX Runtime backend instance with the given configuration string.
func New(config string) (compute.Backend, error) {
	if !IsSupportedPlatform() {
		return nil, errors.Errorf("onnxruntime backend is not supported on platform %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	cudaEnabled, logSeverity, customLibPath, err := parseConfig(config)
	if err != nil {
		return nil, err
	}

	if cudaEnabled {
		if err := cuda.CheckCUDAAndCUDNN(); err != nil {
			return nil, err
		}
	}

	err = initializeORT(cudaEnabled, customLibPath)
	if err != nil {
		return nil, err
	}
	return &Backend{
		config:      config,
		cuda:        cudaEnabled,
		logSeverity: logSeverity,
	}, nil
}

func (b *Backend) createExecutable(modelBytes []byte, inputNames []string, inputShapes []shapes.Shape,
	outputNames []string, outputShapes []shapes.Shape, modelProto *onnx.ModelProto) (compute.Executable, error) {

	session, err := native.CreateSession(modelBytes, inputNames, outputNames, b.cuda, b.logSeverity)
	if err != nil {
		return nil, err
	}
	var savedModelProto *onnx.ModelProto
	if b.keepModelProto {
		savedModelProto = modelProto
	}
	return native.NewExecutable(b, session, inputNames, inputShapes, outputNames, outputShapes, savedModelProto, b.cuda), nil
}

func (b *Backend) BufferFromFlatData(deviceNum compute.DeviceNum, flat any, shape shapes.Shape) (compute.Buffer, error) {
	wrapper, err := native.NewOrtTensorWrapper(shape, flat)
	if err != nil {
		return nil, err
	}
	return native.NewBuffer(b, wrapper, shape, deviceNum, true, false, nil), nil
}

func (b *Backend) HasSharedBuffers() bool {
	return !b.cuda
}

func (b *Backend) NewSharedBuffer(deviceNum compute.DeviceNum, shape shapes.Shape) (compute.Buffer, any, error) {
	wrapper, err := native.NewEmptyOrtTensorWrapper(shape)
	if err != nil {
		return nil, nil, err
	}
	flat := wrapper.GetData()
	buf := native.NewBuffer(b, wrapper, shape, deviceNum, true, false, nil)
	return buf, flat, nil
}
