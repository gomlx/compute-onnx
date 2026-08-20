// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build js && wasm

package onnxbackend

import (
	"fmt"
	"strings"
	"syscall/js"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute-onnx/internal/engine/web"
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

func TestWebNNPresenceCheck(t *testing.T) {
	global := js.Global()
	nav := global.Get("navigator")

	hasRealWebNN := web.HasWebNN()
	t.Logf("Real WebNN presence: %v", hasRealWebNN)

	if !hasRealWebNN {
		// When WebNN is not available, New("webnn") should return a descriptive error
		_, err := New("webnn")
		if err == nil {
			t.Fatal("expected error when requesting webnn in an environment where WebNN is unavailable, got nil")
		}
		if !strings.Contains(err.Error(), "WebNN is not available") || !strings.Contains(err.Error(), "experimental") {
			t.Errorf("unexpected error message: %v", err)
		}
	}

	// Test mock WebNN availability
	originalML := nav.Get("ml")
	defer func() {
		if originalML.IsUndefined() {
			nav.Delete("ml")
		} else {
			nav.Set("ml", originalML)
		}
	}()

	// Mock navigator.ml as an Object
	mockML := global.Get("Object").New()
	nav.Set("ml", mockML)

	if !web.HasWebNN() {
		t.Errorf("expected HasWebNN to be true when navigator.ml is defined")
	}

	// With mock navigator.ml, New("webnn") passes the presence check
	b, err := New("webnn")
	if err != nil {
		t.Fatalf("unexpected error creating backend with mock WebNN: %v", err)
	}
	if b == nil {
		t.Fatal("expected non-nil backend")
	}
	defer b.Finalize()

	if b.Description() != "ONNX Runtime Web (webnn) compute backend for GoMLX" {
		t.Errorf("unexpected description: %s", b.Description())
	}
}
