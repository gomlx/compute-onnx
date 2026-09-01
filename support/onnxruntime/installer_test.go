// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !js

package onnxruntime

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGetDefaultCUDAVersion(t *testing.T) {
	ver := GetDefaultCUDAVersion()
	if ver == "" {
		t.Errorf("GetDefaultCUDAVersion returned empty string")
	}
	t.Logf("Detected default CUDA version: %s", ver)
}

func TestCleanCudaVersion(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"cuda12", "12"},
		{"CUDA12", "12"},
		{"v12", "12"},
		{"12.4", "12"},
		{"cuda12.4.131", "12"},
		{"13", "13"},
		{"", ""},
	}

	for _, tt := range tests {
		got := cleanCudaVersion(tt.input)
		if got != tt.expected {
			t.Errorf("cleanCudaVersion(%q) = %q; want %q", tt.input, got, tt.expected)
		}
	}
}

func TestParseCudaVersionFromOutput(t *testing.T) {
	sampleNvccOut := `nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0`

	ver := parseCudaVersionFromOutput(sampleNvccOut)
	if ver != "12" {
		t.Errorf("parseCudaVersionFromOutput sample nvcc out: got %q; want \"12\"", ver)
	}
}

func TestGetAssetURLWithCudaVersion(t *testing.T) {
	// Test CPU asset URL
	urlCPU, err := GetAssetURL("1.27.0", false, "")
	if err != nil {
		t.Fatalf("GetAssetURL CPU failed: %+v", err)
	}
	if urlCPU == "" {
		t.Errorf("GetAssetURL CPU returned empty URL")
	}

	// Test CUDA asset URL with explicit CUDA version "12"
	urlCUDA12, err := GetAssetURL("1.27.0", true, "12")
	if err != nil {
		t.Fatalf("GetAssetURL CUDA 12 failed: %+v", err)
	}
	if urlCUDA12 == "" {
		t.Errorf("GetAssetURL CUDA 12 returned empty URL")
	}
}

func TestInstallWithCustomTarget(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "ort-install-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %+v", err)
	}
	defer os.RemoveAll(tmpDir)

	targetDir := filepath.Join(tmpDir, "custom_ort")
	libPath, err := Install("1.27.0", false, "", targetDir, false)
	if err != nil {
		t.Fatalf("Install to custom target failed: %+v", err)
	}

	expectedLib, err := GetLibFilename()
	if err != nil {
		t.Fatalf("GetLibFilename failed: %+v", err)
	}
	expectedPath := filepath.Join(targetDir, expectedLib)

	if libPath != expectedPath {
		t.Errorf("Install returned libPath %q; want %q", libPath, expectedPath)
	}

	if _, err := os.Stat(libPath); err != nil {
		t.Errorf("Installed library does not exist at %q", libPath)
	}
}

func TestGetLatestVersion(t *testing.T) {
	ver, err := GetLatestVersion(true)
	if err != nil {
		t.Fatalf("GetLatestVersion(true) failed: %+v", err)
	}
	t.Logf("GetLatestVersion(true) returned: %s", ver)
	if isCUDAVersionLE12_4(GetCUDAFullVersion()) && ver != "1.27.0" {
		t.Errorf("GetLatestVersion(true) = %q; want \"1.27.0\" on CUDA <= 12.4", ver)
	}
}

func TestNormalizeVersion(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"1.29", "1.29.0"},
		{"v1.29", "1.29.0"},
		{"V1.29", "1.29.0"},
		{"1.29.0", "1.29.0"},
		{"v1.29.0", "1.29.0"},
		{"1.27", "1.27.1"},
		{"v1.24", "1.24.4"},
		{"", ""},
	}

	for _, tt := range tests {
		got := NormalizeVersion(tt.input)
		if got != tt.expected {
			t.Errorf("NormalizeVersion(%q) = %q; want %q", tt.input, got, tt.expected)
		}
	}
}

func TestFindMigraphxWheelURL(t *testing.T) {
	url, err := FindMigraphxWheelURL("7.2.4")
	if err != nil {
		t.Skipf("network unavailable or listing changed: %+v", err)
	}
	if !strings.Contains(url, "rocm-rel-7.2.4/") ||
		!strings.Contains(url, "onnxruntime_migraphx-") ||
		!strings.HasSuffix(url, ".whl") {
		t.Errorf("unexpected wheel URL: %q", url)
	}
}

func TestInstallMigraphxWithCustomTarget(t *testing.T) {
	targetDir := t.TempDir()
	libPath, err := InstallMigraphx("7.2.4", targetDir, true)
	if err != nil {
		t.Skipf("network unavailable or install failed: %+v", err)
	}
	if libPath != filepath.Join(targetDir, "libonnxruntime.so") {
		t.Errorf("unexpected library path returned: %q", libPath)
	}
	for _, name := range []string{
		"libonnxruntime_providers_migraphx.so",
		"libonnxruntime_providers_shared.so",
	} {
		if _, err := os.Stat(filepath.Join(targetDir, name)); err != nil {
			t.Errorf("expected %s to be installed: %+v", name, err)
		}
	}
	// Idempotent install (no force).
	if _, err := InstallMigraphx("7.2.4", targetDir, false); err != nil {
		t.Errorf("second non-forced install should be a no-op success, got: %+v", err)
	}
}
