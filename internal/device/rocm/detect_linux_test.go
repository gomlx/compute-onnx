// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build linux

package rocm

import (
	"os"
	"path/filepath"
	"testing"
)

const sampleRocminfo = `===================== ROCm System Management Interface ================================
=========================== HSA System Agent Report  ==================================
*******                1      *******
  Name:                    AMD Ryzen 9 7900X 12-Core Processor
  Uuid:                    CPU-None
  Marketing Name:          AMD Ryzen 9 7900X 12-Core Processor
  Vendor Name:             CPU
  Feature:                 None specified
  Pool Info:
    Pool 1
      Segment:                 GLOBAL; FLAGS: FINE GRAINED
      Size:                    65790208(62742 KB)
  Cache Info:
    L1:                      64(64 KB)
*******                2      *******
  Name:                    gfx1100
  Uuid:                    GPU-XX
  Marketing Name:          AMD Radeon RX 7900 XTX
  Vendor Name:             AMD
  Feature:                 KERNEL_DISPATCH
  Profile:                 BASE_PROFILE
  Peak Number of Compute Units:      96
  Pool Info:
    Pool 1
      Segment:                 GLOBAL; FLAGS:
      Memory Properties:       DDR4,HBM,SRAM
      Size:                    25756528640(24563 MB)
*******                3      *******
  Name:                    gfx940
  Uuid:                    GPU-YY
  Marketing Name:          AMD APU Graphics
  Vendor Name:             AMD
  Feature:                 KERNEL_DISPATCH
  Pool Info:
    Pool 1
      Segment:                 GLOBAL; FLAGS:
      Memory Properties:       APU
      Size:                    1024(1024 MB)
=======================================================================================
====================End of ROCm SMI Log ==============================================`

const cpuOnlyRocminfo = `*******                1      *******
  Name:                    AMD Ryzen 9 7900X 12-Core Processor
  Vendor Name:             CPU
*******`

func TestRocminfoHasDiscreteGPU(t *testing.T) {
	if !rocminfoHasDiscreteGPU(sampleRocminfo) {
		t.Errorf("expected sample with RX 7900 XTX to report a discrete GPU")
	}
	if rocminfoHasDiscreteGPU(cpuOnlyRocminfo) {
		t.Errorf("expected CPU-only rocminfo output to not report a discrete GPU")
	}
}

func TestRocminfoField(t *testing.T) {
	block := `
  Name:                    gfx1100
    Marketing Name:        AMD Radeon RX 7900 XTX`
	if got := rocminfoField(block, "Name:"); got != "gfx1100" {
		t.Errorf("rocminfoField(Name:) = %q, want %q", got, "gfx1100")
	}
	if got := rocminfoField(block, "Marketing Name:"); got != "AMD Radeon RX 7900 XTX" {
		t.Errorf("rocminfoField(Marketing Name:) = %q, want %q", got, "AMD Radeon RX 7900 XTX")
	}
	if got := rocminfoField(block, "Uuid:"); got != "" {
		t.Errorf("rocminfoField(Uuid:) = %q, want empty", got)
	}
}

func TestHasMigraphxExecutionProvider(t *testing.T) {
	dir := t.TempDir()
	if HasMigraphxExecutionProvider(dir) {
		t.Errorf("expected empty directory to not have migraphx provider library")
	}
	f, err := os.Create(filepath.Join(dir, "libonnxruntime_providers_migraphx.so"))
	if err != nil {
		t.Fatalf("failed to create fake provider library: %+v", err)
	}
	f.Close()
	if !HasMigraphxExecutionProvider(dir) {
		t.Errorf("expected directory with libonnxruntime_providers_migraphx.so to be detected")
	}
}
