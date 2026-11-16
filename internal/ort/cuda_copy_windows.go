// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build windows

package ort

/*
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stddef.h>

// cudaMemcpyKind enum values
#define CUDA_MEMCPY_HOST_TO_DEVICE 1
#define CUDA_MEMCPY_DEVICE_TO_HOST 2

typedef int (*cudaMemcpyFn)(void* dst, const void* src, size_t count, int kind);

static cudaMemcpyFn resolved_cudaMemcpy = NULL;

// Resolve cudaMemcpy from an already-loaded or standard Windows CUDA runtime DLL.
// Returns 0 on success, -1 if the symbol cannot be found.
static int resolve_cuda_symbols() {
    if (resolved_cudaMemcpy != NULL) return 0;

    const char* names[] = {"cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll", "cudart64_11.dll"};
    for (int i = 0; i < 4 && resolved_cudaMemcpy == NULL; i++) {
        HMODULE mod = GetModuleHandleA(names[i]);
        if (mod == NULL) {
            mod = LoadLibraryA(names[i]);
        }
        if (mod != NULL) {
            resolved_cudaMemcpy = (cudaMemcpyFn)GetProcAddress(mod, "cudaMemcpy");
        }
    }
    if (resolved_cudaMemcpy == NULL) return -1;
    return 0;
}

static int cuda_memcpy_d2h(void* dst, const void* src, size_t count) {
    if (resolve_cuda_symbols() != 0) return -1;
    return resolved_cudaMemcpy(dst, src, count, CUDA_MEMCPY_DEVICE_TO_HOST);
}

static int cuda_memcpy_h2d(void* dst, const void* src, size_t count) {
    if (resolve_cuda_symbols() != 0) return -1;
    return resolved_cudaMemcpy(dst, src, count, CUDA_MEMCPY_HOST_TO_DEVICE);
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// CudaMemcpyD2H copies byteSize bytes from a CUDA device pointer to a host pointer.
// It resolves cudaMemcpy from the Windows CUDA runtime DLL.
func CudaMemcpyD2H(hostDst unsafe.Pointer, deviceSrc unsafe.Pointer, byteSize int) error {
	ret := C.cuda_memcpy_d2h(hostDst, deviceSrc, C.size_t(byteSize))
	if ret != 0 {
		return fmt.Errorf("cudaMemcpy D2H failed with error code %d", int(ret))
	}
	return nil
}

// CudaMemcpyH2D copies byteSize bytes from a host pointer to a CUDA device pointer.
// It resolves cudaMemcpy from the Windows CUDA runtime DLL.
func CudaMemcpyH2D(deviceDst unsafe.Pointer, hostSrc unsafe.Pointer, byteSize int) error {
	ret := C.cuda_memcpy_h2d(deviceDst, hostSrc, C.size_t(byteSize))
	if ret != 0 {
		return fmt.Errorf("cudaMemcpy H2D failed with error code %d", int(ret))
	}
	return nil
}
