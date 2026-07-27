// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ort

/*
#include <dlfcn.h>
#include <stddef.h>
#include <string.h>

// cudaMemcpyKind enum value for DeviceToHost
#define CUDA_MEMCPY_DEVICE_TO_HOST 2

typedef int (*cudaMemcpyFn)(void* dst, const void* src, size_t count, int kind);
typedef int (*cudaDeviceSynchronizeFn)(void);

static cudaMemcpyFn resolved_cudaMemcpy = NULL;
static cudaDeviceSynchronizeFn resolved_cudaDeviceSync = NULL;

// Resolve cudaMemcpy — try RTLD_DEFAULT first, then explicitly load libcudart.
// Returns 0 on success, -1 if the symbol cannot be found.
static int resolve_cuda_symbols() {
    if (resolved_cudaMemcpy != NULL) return 0;

    // Try 1: already in global symbol table.
    resolved_cudaMemcpy = (cudaMemcpyFn)dlsym(RTLD_DEFAULT, "cudaMemcpy");
    if (resolved_cudaMemcpy == NULL) {
        // Try 2: explicitly load libcudart.so.
        void* handle = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
        if (handle == NULL) {
            // Try versioned name.
            handle = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
        }
        if (handle == NULL) {
            handle = dlopen("libcudart.so.11", RTLD_NOW | RTLD_GLOBAL);
        }
        if (handle != NULL) {
            resolved_cudaMemcpy = (cudaMemcpyFn)dlsym(handle, "cudaMemcpy");
        }
    }
    if (resolved_cudaMemcpy == NULL) return -1;

    resolved_cudaDeviceSync = (cudaDeviceSynchronizeFn)dlsym(RTLD_DEFAULT, "cudaDeviceSynchronize");
    if (resolved_cudaDeviceSync == NULL) {
        // Try from the handle we just opened.
        void* handle = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
        if (handle != NULL) {
            resolved_cudaDeviceSync = (cudaDeviceSynchronizeFn)dlsym(handle, "cudaDeviceSynchronize");
        }
    }
    return 0;
}

// Copy from GPU device memory to host memory.
// Returns CUDA error code (0 = success).
static int cuda_memcpy_d2h(void* dst, const void* src, size_t count) {
    if (resolve_cuda_symbols() != 0) return -1;
    // cudaMemcpy with DeviceToHost is synchronous on the default stream.
    return resolved_cudaMemcpy(dst, src, count, CUDA_MEMCPY_DEVICE_TO_HOST);
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// CudaMemcpyD2H copies byteSize bytes from a CUDA device pointer to a host pointer.
// It uses dlsym to resolve cudaMemcpy from the already-loaded CUDA runtime.
func CudaMemcpyD2H(hostDst unsafe.Pointer, deviceSrc unsafe.Pointer, byteSize int) error {
	ret := C.cuda_memcpy_d2h(hostDst, deviceSrc, C.size_t(byteSize))
	if ret != 0 {
		return fmt.Errorf("cudaMemcpy D2H failed with error code %d", int(ret))
	}
	return nil
}
