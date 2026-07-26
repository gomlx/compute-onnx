// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ort

/*
#include "onnxruntime_c_api.h"
#include <stdlib.h>
#include <string.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type Arena struct {
	buf     []byte
	current int
	size    int
}

func NewArena(size int) *Arena {
	ptr := C.calloc(C.size_t(size), 1)
	return &Arena{
		buf:  unsafe.Slice((*byte)(ptr), size),
		size: size,
	}
}

func (a *Arena) Free() {
	if len(a.buf) > 0 {
		C.free(unsafe.Pointer(&a.buf[0]))
		a.buf = nil
		a.size = 0
		a.current = 0
	}
}

func (a *Arena) AllocInt64Slice(n int) *C.int64_t {
	allocSize := n * 8
	if a.current+allocSize > a.size {
		panic(fmt.Sprintf("Arena out of memory: current=%d, requested=%d, size=%d", a.current, allocSize, a.size))
	}
	ptr := unsafe.Pointer(&a.buf[a.current])
	a.current += allocSize
	a.current = (a.current + 7) &^ 7
	return (*C.int64_t)(ptr)
}

func (a *Arena) AllocCharStarSlice(n int) **C.char {
	sizeOfPtr := int(unsafe.Sizeof(uintptr(0)))
	allocSize := n * sizeOfPtr
	if a.current+allocSize > a.size {
		panic(fmt.Sprintf("Arena out of memory: current=%d, requested=%d, size=%d", a.current, allocSize, a.size))
	}
	ptr := unsafe.Pointer(&a.buf[a.current])
	a.current += allocSize
	a.current = (a.current + 7) &^ 7
	return (**C.char)(ptr)
}

func (a *Arena) AllocOrtValueStarSlice(n int) **C.OrtValue {
	sizeOfPtr := int(unsafe.Sizeof(uintptr(0)))
	allocSize := n * sizeOfPtr
	if a.current+allocSize > a.size {
		panic(fmt.Sprintf("Arena out of memory: current=%d, requested=%d, size=%d", a.current, allocSize, a.size))
	}
	ptr := unsafe.Pointer(&a.buf[a.current])
	a.current += allocSize
	a.current = (a.current + 7) &^ 7
	return (**C.OrtValue)(ptr)
}

func (a *Arena) AllocCString(s string) *C.char {
	n := len(s)
	allocSize := n + 1
	if a.current+allocSize > a.size {
		panic(fmt.Sprintf("Arena out of memory: current=%d, requested=%d, size=%d", a.current, allocSize, a.size))
	}
	ptr := unsafe.Pointer(&a.buf[a.current])
	a.current += allocSize
	// Copy string bytes with null terminator
	dst := unsafe.Slice((*byte)(ptr), allocSize)
	copy(dst, s)
	dst[n] = 0 // null terminator
	return (*C.char)(ptr)
}
