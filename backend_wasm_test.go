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
	"github.com/gomlx/compute-onnx/internal/executionprovider"
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

	if !strings.HasPrefix(b.Description(), "ONNX Runtime Web") || !strings.HasSuffix(b.Description(), "(webnn) compute backend for GoMLX") {
		t.Errorf("unexpected description: %s", b.Description())
	}
}

func TestParseConfigGraphCapture(t *testing.T) {
	tests := []struct {
		config           string
		wantEP           executionprovider.Type
		wantGraphCapture bool
		wantWebVersion   string
		wantErr          bool
	}{
		{config: "", wantEP: executionprovider.WebGPU, wantGraphCapture: false, wantWebVersion: ""},
		{config: "wasm", wantEP: executionprovider.WASM, wantGraphCapture: false, wantWebVersion: ""},
		{config: "webgpu", wantEP: executionprovider.WebGPU, wantGraphCapture: false, wantWebVersion: ""},
		{config: "webgpu,graph_capture=true", wantEP: executionprovider.WebGPU, wantGraphCapture: true, wantWebVersion: ""},
		{config: "webgpu,graph_capture=1", wantEP: executionprovider.WebGPU, wantGraphCapture: true, wantWebVersion: ""},
		{config: "webgpu,graph_capture=false", wantEP: executionprovider.WebGPU, wantGraphCapture: false, wantWebVersion: ""},
		{config: "webgpu,graph_capture=0", wantEP: executionprovider.WebGPU, wantGraphCapture: false, wantWebVersion: ""},
		{config: "webgpu,graph_capture", wantEP: executionprovider.WebGPU, wantGraphCapture: true, wantWebVersion: ""},
		{config: "onnx:webgpu,graph_capture=true,log=2", wantEP: executionprovider.WebGPU, wantGraphCapture: true, wantWebVersion: ""},
		{config: "wasm,web_version=1.27", wantEP: executionprovider.WASM, wantGraphCapture: false, wantWebVersion: "1.27"},
		{config: "webgpu,web_version=@latest", wantEP: executionprovider.WebGPU, wantGraphCapture: false, wantWebVersion: "@latest"},
		{config: "webgpu,webversion=v1.27", wantEP: executionprovider.WebGPU, wantGraphCapture: false, wantWebVersion: "v1.27"},
		{config: "webgpu,graph_capture=invalid_bool", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.config, func(t *testing.T) {
			ep, _, gc, wv, err := parseConfig(tt.config)
			if (err != nil) != tt.wantErr {
				t.Fatalf("parseConfig(%q) error = %v, wantErr %v", tt.config, err, tt.wantErr)
			}
			if !tt.wantErr {
				if ep != tt.wantEP && !(tt.config == "" && !web.HasWebGPU() && ep == executionprovider.WASM) {
					t.Errorf("ep = %v, want %v", ep, tt.wantEP)
				}
				if gc != tt.wantGraphCapture {
					t.Errorf("graphCapture = %v, want %v", gc, tt.wantGraphCapture)
				}
				if wv != tt.wantWebVersion {
					t.Errorf("webVersion = %v, want %v", wv, tt.wantWebVersion)
				}
			}
		})
	}
}

func TestNormalizeORTWebVersion(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"", "dev"},
		{"@dev", "dev"},
		{"dev", "dev"},
		{"@latest", "latest"},
		{"latest", "latest"},
		{"1.27", "1.27"},
		{"v1.27", "1.27"},
		{"V1.27.0", "1.27.0"},
	}
	for _, tc := range cases {
		got := web.NormalizeORTWebVersion(tc.in)
		if got != tc.want {
			t.Errorf("NormalizeORTWebVersion(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
