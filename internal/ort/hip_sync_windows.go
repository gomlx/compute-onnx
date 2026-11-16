// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build windows

package ort

import "github.com/pkg/errors"

// HipDeviceSynchronize is not supported on Windows.
func HipDeviceSynchronize() error {
	return errors.New("hipDeviceSynchronize is not supported on Windows")
}

// LoadHIPLibrary is not supported on Windows and always returns false.
func LoadHIPLibrary() bool {
	return false
}
