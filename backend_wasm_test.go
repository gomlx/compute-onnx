// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build js && wasm

package onnxbackend

import (
	"fmt"
	"os"
	"strings"
	"syscall/js"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute-onnx/internal/engine/web"
	"k8s.io/klog/v2"
)

func setup() {
	fmt.Printf("Available backends (WASM): %q\n", compute.List())
	backendName := os.Getenv("GOMLX_BACKEND")
	var errNew error
	if backendName == "" {
		backend, errNew = compute.New()
	} else {
		backend, errNew = compute.NewWithConfig(backendName)
	}
	if errNew != nil {
		klog.Fatalf("Failed to create backend (%q): %+v", backendName, errNew)
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

func TestParseConfigGraphCapture(t *testing.T) {
	tests := []struct {
		config             string
		wantEP             string
		wantGraphCapture   bool
		wantErr            bool
	}{
		{config: "", wantEP: "webgpu", wantGraphCapture: false},
		{config: "wasm", wantEP: "wasm", wantGraphCapture: false},
		{config: "webgpu", wantEP: "webgpu", wantGraphCapture: false},
		{config: "webgpu,graph_capture=true", wantEP: "webgpu", wantGraphCapture: true},
		{config: "webgpu,graph_capture=1", wantEP: "webgpu", wantGraphCapture: true},
		{config: "webgpu,graph_capture=false", wantEP: "webgpu", wantGraphCapture: false},
		{config: "webgpu,graph_capture=0", wantEP: "webgpu", wantGraphCapture: false},
		{config: "webgpu,graph_capture", wantEP: "webgpu", wantGraphCapture: true},
		{config: "onnx:webgpu,graph_capture=true,log=2", wantEP: "webgpu", wantGraphCapture: true},
		{config: "webgpu,graph_capture=invalid_bool", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.config, func(t *testing.T) {
			ep, _, gc, err := parseConfig(tt.config)
			if (err != nil) != tt.wantErr {
				t.Fatalf("parseConfig(%q) error = %v, wantErr %v", tt.config, err, tt.wantErr)
			}
			if !tt.wantErr {
				if ep != tt.wantEP && !(tt.config == "" && !web.HasWebGPU() && ep == "wasm") {
					t.Errorf("ep = %v, want %v", ep, tt.wantEP)
				}
				if gc != tt.wantGraphCapture {
					t.Errorf("graphCapture = %v, want %v", gc, tt.wantGraphCapture)
				}
			}
		})
	}
}
