// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package native

import (
	"os"
	"strings"

	ort "github.com/gomlx/compute-onnx/internal/ort"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// SaveOnFailureEnv is the environment variable that, when set to a file path,
// instructs the ONNX Runtime backend to save the serialized ONNX model protobuf to
// that path if graph compilation / session creation fails.
const SaveOnFailureEnv = "GOMLX_ONNX_SAVE_ON_FAILURE"

// CreateSession creates an ONNX Runtime DynamicAdvancedSession with the given options.
func CreateSession(modelBytes []byte, inputNames []string, outputNames []string, cuda bool, logSeverity int) (*ort.DynamicAdvancedSession, error) {
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, errors.Wrap(err, "failed to create ONNX Runtime SessionOptions")
	}
	defer options.Destroy()

	if cuda {
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return nil, errors.Wrap(err, "failed to create ONNX Runtime CUDAProviderOptions")
		}
		defer cudaOpts.Destroy()

		_ = cudaOpts.Update(map[string]string{
			"do_copy_in_default_stream": "1",
		})

		err = options.AppendExecutionProviderCUDA(cudaOpts)
		if err != nil {
			return nil, WrapCUDAError(errors.Wrap(err, "failed to append CUDA execution provider to SessionOptions"))
		}
	}

	logSev := logSeverity
	if logSev < 0 {
		logSev = 3 // ORT_LOGGING_LEVEL_ERROR by default
	}
	err = options.SetSessionLogSeverityLevel(logSev)
	if err != nil {
		return nil, errors.Wrap(err, "failed to set ONNX Runtime session log severity level")
	}

	session, err := ort.NewDynamicAdvancedSessionWithONNXData(modelBytes, inputNames, outputNames, options)
	if err != nil {
		if filePath := os.Getenv(SaveOnFailureEnv); filePath != "" {
			if wErr := os.WriteFile(filePath, modelBytes, 0644); wErr == nil {
				klog.Infof("Saving failed model to %q ($%s)", filePath, SaveOnFailureEnv)
			} else {
				klog.Errorf("Failed to save failed model to %q ($%s): %+v", filePath, SaveOnFailureEnv, wErr)
			}
		}
		return nil, WrapCUDAError(errors.Wrap(err, "failed to create ONNX Runtime session"))
	}

	return session, nil
}

// WrapCUDAError adds helpful troubleshooting messages to known CUDA / ORT initialization errors.
func WrapCUDAError(err error) error {
	if err == nil {
		return nil
	}
	errStr := err.Error()
	if strings.Contains(errStr, "cudaLibraryGetKernel") ||
		strings.Contains(errStr, "OrtSessionOptionsAppendExecutionProvider_Cuda: Failed to load shared library") ||
		(strings.Contains(errStr, "AppendExecutionProvider_Cuda") && strings.Contains(errStr, "Failed to load")) {
		return errors.WithMessage(err, "CUDA versions <= 12.4 are compatible only with ORT (ONNX Runtime) up to v1.27. Maybe install an earlier version of ORT ? (see github.com/gomlx/compute-onnx/cmd/onnxruntime_installer)")
	}
	return err
}
