// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build js && wasm

package onnxbackend

import (
	"fmt"

	"github.com/gomlx/compute"
	"k8s.io/klog/v2"
)

func setup() {
	fmt.Printf("Available backends (WASM): %q\n", compute.List())
	var errNew error
	backend, errNew = New("")
	if errNew != nil {
		klog.Fatalf("Failed to create backend: %+v", errNew)
	}
	fmt.Printf("Backend: %s, %s\n", backend.Name(), backend.Description())
}

func teardown() {
	if backend != nil {
		backend.Finalize()
	}
}
