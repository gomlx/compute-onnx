//go:build !linux

package onnxbackend

// HasNvidiaGPU returns false on non-Linux platforms.
func HasNvidiaGPU() bool {
	return false
}

// checkCUDAAndCUDNN returns nil on non-Linux platforms.
func checkCUDAAndCUDNN() error {
	return nil
}
