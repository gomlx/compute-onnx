// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build windows

package ort

/*
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdlib.h>

static void* win_load_library(const wchar_t* path, DWORD* err_code) {
	HMODULE handle = LoadLibraryExW(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
	if (handle == NULL) {
		handle = LoadLibraryW(path);
	}
	if (handle == NULL) {
		*err_code = GetLastError();
		return NULL;
	}
	*err_code = 0;
	return (void*)handle;
}

static void* win_get_proc_address(void* handle, const char* name, DWORD* err_code) {
	FARPROC proc = GetProcAddress((HMODULE)handle, name);
	if (proc == NULL) {
		*err_code = GetLastError();
		return NULL;
	}
	*err_code = 0;
	return (void*)proc;
}

static int win_free_library(void* handle, DWORD* err_code) {
	if (!FreeLibrary((HMODULE)handle)) {
		*err_code = GetLastError();
		return -1;
	}
	*err_code = 0;
	return 0;
}
*/
import "C"
import (
	"fmt"
	"path/filepath"
	"syscall"
	"unsafe"
)

func loadLibrary(libPath string) (unsafe.Pointer, error) {
	winPath := filepath.FromSlash(libPath)
	pathUTF16, err := syscall.UTF16PtrFromString(winPath)
	if err != nil {
		return nil, fmt.Errorf("invalid library path %q: %w", libPath, err)
	}

	var errCode C.DWORD
	handle := C.win_load_library((*C.wchar_t)(unsafe.Pointer(pathUTF16)), &errCode)
	if handle == nil {
		return nil, fmt.Errorf("failed to LoadLibrary %q: %w", libPath, syscall.Errno(errCode))
	}
	return handle, nil
}

func getOrtGetApiBase(handle unsafe.Pointer) (unsafe.Pointer, error) {
	symC := C.CString("OrtGetApiBase")
	defer C.free(unsafe.Pointer(symC))

	var errCode C.DWORD
	p := C.win_get_proc_address(handle, symC, &errCode)
	if p == nil {
		return nil, fmt.Errorf("failed to find symbol OrtGetApiBase: %w", syscall.Errno(errCode))
	}
	return p, nil
}

func closeLibrary(handle unsafe.Pointer) error {
	var errCode C.DWORD
	if C.win_free_library(handle, &errCode) != 0 {
		return fmt.Errorf("failed to FreeLibrary: %w", syscall.Errno(errCode))
	}
	return nil
}
