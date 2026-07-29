// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"flag"
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/support/backendtest"
	"k8s.io/klog/v2"
)

var (
	backend compute.Backend
)

func TestCompliance(t *testing.T) {
	backendtest.RunAll(t, backend, nil)
}

func init() {
	klog.InitFlags(nil)
}

func setup() {
	fmt.Printf("Available backends: %q\n", compute.List())
	var err error
	backend, err = New("")
	if err != nil {
		klog.Fatalf("Failed to create backend: %+v", err)
	}
	fmt.Printf("Backend: %s, %s\n", backend.Name(), backend.Description())
}

func teardown() {
	if backend != nil {
		backend.Finalize()
	}
}

func TestMain(m *testing.M) {
	flag.Parse()
	setup()
	code := m.Run()
	teardown()
	os.Exit(code)
}

