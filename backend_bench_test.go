// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxbackend

import (
	"fmt"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

func BenchmarkTransfer(b *testing.B) {
	sizes := []int{10, 1000, 1_000_000}

	// Breakdown benchmark for 10 floats to see exactly where time goes
	b.Run("Breakdown_Size10", func(b *testing.B) {
		builder := backend.Builder("breakdown_10")
		fn := builder.Main()
		shape := shapes.Make(dtypes.Float32, 10)

		xParam, err := fn.Parameter("x", shape, nil)
		if err != nil {
			b.Fatalf("failed to create param x: %+v", err)
		}
		oneScalar, err := fn.Constant([]float32{1.0})
		if err != nil {
			b.Fatalf("failed to create scalar 1.0: %+v", err)
		}
		outVal, err := fn.Add(xParam, oneScalar)
		if err != nil {
			b.Fatalf("failed to create Add: %+v", err)
		}
		fn.Return([]compute.Value{outVal}, nil)

		exec, err := builder.Compile()
		if err != nil {
			b.Fatalf("failed to compile: %+v", err)
		}
		defer exec.Finalize()

		xData := make([]float32, 10)
		outData := make([]float32, 10)

		// Measure BufferFromFlatData alone
		b.Run("1_BufferFromFlatData", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				inBuf, _ := backend.BufferFromFlatData(0, xData, shape)
				_ = inBuf.Finalize()
			}
		})

		// Measure Execute alone (pre-created buffer)
		b.Run("2_Execute_PrecreatedBuffer", func(b *testing.B) {
			inBuf, _ := backend.BufferFromFlatData(0, xData, shape)
			defer inBuf.Finalize()

			for i := 0; i < b.N; i++ {
				outBufs, errExec := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
				if errExec != nil {
					b.Fatalf("Execute failed: %+v", errExec)
				}
				_ = outBufs[0].Finalize()
			}
		})

		// Measure Execute + ToFlatData
		b.Run("3_Execute_Plus_ToFlatData", func(b *testing.B) {
			inBuf, _ := backend.BufferFromFlatData(0, xData, shape)
			defer inBuf.Finalize()

			for i := 0; i < b.N; i++ {
				outBufs, errExec := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
				if errExec != nil {
					b.Fatalf("Execute failed: %+v", errExec)
				}
				_ = outBufs[0].ToFlatData(outData)
				_ = outBufs[0].Finalize()
			}
		})
	})

	// Breakdown benchmark for 4 inputs of Size 10
	b.Run("Breakdown_4Inputs_Size10", func(b *testing.B) {
		builder := backend.Builder("breakdown_4_10")
		fn := builder.Main()
		shape := shapes.Make(dtypes.Float32, 10)

		aParam, _ := fn.Parameter("a", shape, nil)
		bParam, _ := fn.Parameter("b", shape, nil)
		cParam, _ := fn.Parameter("c", shape, nil)
		dParam, _ := fn.Parameter("d", shape, nil)

		ab, _ := fn.Add(aParam, bParam)
		abc, _ := fn.Add(ab, cParam)
		abcd, _ := fn.Add(abc, dParam)
		sumVal, _ := fn.ReduceSum(abcd)
		fn.Return([]compute.Value{sumVal}, nil)

		exec, err := builder.Compile()
		if err != nil {
			b.Fatalf("failed to compile: %+v", err)
		}
		defer exec.Finalize()

		dataSlice := make([]float32, 10)
		for i := range dataSlice {
			dataSlice[i] = 1.0
		}
		outData := make([]float32, 1)

		b.Run("1_Execute_PrecreatedBuffers", func(b *testing.B) {
			inA, _ := backend.BufferFromFlatData(0, dataSlice, shape)
			inB, _ := backend.BufferFromFlatData(0, dataSlice, shape)
			inC, _ := backend.BufferFromFlatData(0, dataSlice, shape)
			inD, _ := backend.BufferFromFlatData(0, dataSlice, shape)
			defer inA.Finalize()
			defer inB.Finalize()
			defer inC.Finalize()
			defer inD.Finalize()

			for i := 0; i < b.N; i++ {
				outBufs, errExec := exec.Execute([]compute.Buffer{inA, inB, inC, inD}, []bool{false, false, false, false}, 0)
				if errExec != nil {
					b.Fatalf("Execute failed: %+v", errExec)
				}
				_ = outBufs[0].Finalize()
			}
		})

		b.Run("2_Execute_Plus_ToFlatData", func(b *testing.B) {
			inA, _ := backend.BufferFromFlatData(0, dataSlice, shape)
			inB, _ := backend.BufferFromFlatData(0, dataSlice, shape)
			inC, _ := backend.BufferFromFlatData(0, dataSlice, shape)
			inD, _ := backend.BufferFromFlatData(0, dataSlice, shape)
			defer inA.Finalize()
			defer inB.Finalize()
			defer inC.Finalize()
			defer inD.Finalize()

			for i := 0; i < b.N; i++ {
				outBufs, errExec := exec.Execute([]compute.Buffer{inA, inB, inC, inD}, []bool{false, false, false, false}, 0)
				if errExec != nil {
					b.Fatalf("Execute failed: %+v", errExec)
				}
				_ = outBufs[0].ToFlatData(outData)
				_ = outBufs[0].Finalize()
			}
		})
	})

	// 1. Benchmark f(x) = x + 1
	b.Run("AddScalar_f(x)=x+1", func(b *testing.B) {
		for _, size := range sizes {
			b.Run(fmt.Sprintf("Size=%d", size), func(b *testing.B) {
				builder := backend.Builder(fmt.Sprintf("add_scalar_%d", size))
				fn := builder.Main()
				shape := shapes.Make(dtypes.Float32, size)

				xParam, err := fn.Parameter("x", shape, nil)
				if err != nil {
					b.Fatalf("failed to create param x: %+v", err)
				}
				oneScalar, err := fn.Constant([]float32{1.0})
				if err != nil {
					b.Fatalf("failed to create scalar 1.0: %+v", err)
				}
				outVal, err := fn.Add(xParam, oneScalar)
				if err != nil {
					b.Fatalf("failed to create Add: %+v", err)
				}
				fn.Return([]compute.Value{outVal}, nil)

				exec, err := builder.Compile()
				if err != nil {
					b.Fatalf("failed to compile: %+v", err)
				}
				defer exec.Finalize()

				// Prepare input data
				xData := make([]float32, size)
				for i := range xData {
					xData[i] = float32(i)
				}
				outData := make([]float32, size)

				// Warmup runs
				for w := 0; w < 3; w++ {
					inBuf, errBuf := backend.BufferFromFlatData(0, xData, shape)
					if errBuf != nil {
						b.Fatalf("BufferFromFlatData failed: %+v", errBuf)
					}
					outBufs, errExec := exec.Execute([]compute.Buffer{inBuf}, []bool{true}, 0)
					if errExec != nil {
						b.Fatalf("Execute failed: %+v", errExec)
					}
					_ = outBufs[0].ToFlatData(outData)
					_ = outBufs[0].Finalize()
				}

				b.ResetTimer()
				b.SetBytes(int64(size * 4 * 2)) // input + output float32 bytes
				for i := 0; i < b.N; i++ {
					inBuf, errBuf := backend.BufferFromFlatData(0, xData, shape)
					if errBuf != nil {
						b.Fatalf("BufferFromFlatData failed: %+v", errBuf)
					}
					outBufs, errExec := exec.Execute([]compute.Buffer{inBuf}, []bool{true}, 0)
					if errExec != nil {
						b.Fatalf("Execute failed: %+v", errExec)
					}
					errRead := outBufs[0].ToFlatData(outData)
					if errRead != nil {
						b.Fatalf("ToFlatData failed: %+v", errRead)
					}
					_ = outBufs[0].Finalize()
				}
				b.StopTimer()
			})
		}
	})

	// 2. Benchmark f(a, b, c, d) = ReduceSum(a + b + c + d)
	b.Run("ReduceSum_f(a,b,c,d)=Sum(a+b+c+d)", func(b *testing.B) {
		for _, size := range sizes {
			b.Run(fmt.Sprintf("Size=%d", size), func(b *testing.B) {
				builder := backend.Builder(fmt.Sprintf("reduce_sum_abcd_%d", size))
				fn := builder.Main()
				shape := shapes.Make(dtypes.Float32, size)

				aParam, err := fn.Parameter("a", shape, nil)
				if err != nil {
					b.Fatalf("failed to create param a: %+v", err)
				}
				bParam, err := fn.Parameter("b", shape, nil)
				if err != nil {
					b.Fatalf("failed to create param b: %+v", err)
				}
				cParam, err := fn.Parameter("c", shape, nil)
				if err != nil {
					b.Fatalf("failed to create param c: %+v", err)
				}
				dParam, err := fn.Parameter("d", shape, nil)
				if err != nil {
					b.Fatalf("failed to create param d: %+v", err)
				}

				ab, err := fn.Add(aParam, bParam)
				if err != nil {
					b.Fatalf("failed to Add a+b: %+v", err)
				}
				abc, err := fn.Add(ab, cParam)
				if err != nil {
					b.Fatalf("failed to Add ab+c: %+v", err)
				}
				abcd, err := fn.Add(abc, dParam)
				if err != nil {
					b.Fatalf("failed to Add abc+d: %+v", err)
				}

				sumVal, err := fn.ReduceSum(abcd)
				if err != nil {
					b.Fatalf("failed to ReduceSum: %+v", err)
				}
				fn.Return([]compute.Value{sumVal}, nil)

				exec, err := builder.Compile()
				if err != nil {
					b.Fatalf("failed to compile: %+v", err)
				}
				defer exec.Finalize()

				// Prepare input data
				dataSlice := make([]float32, size)
				for i := range dataSlice {
					dataSlice[i] = 1.0
				}
				outData := make([]float32, 1)

				// Warmup runs
				for w := 0; w < 3; w++ {
					inA, _ := backend.BufferFromFlatData(0, dataSlice, shape)
					inB, _ := backend.BufferFromFlatData(0, dataSlice, shape)
					inC, _ := backend.BufferFromFlatData(0, dataSlice, shape)
					inD, _ := backend.BufferFromFlatData(0, dataSlice, shape)
					outBufs, errExec := exec.Execute([]compute.Buffer{inA, inB, inC, inD}, []bool{true, true, true, true}, 0)
					if errExec != nil {
						b.Fatalf("Execute failed: %+v", errExec)
					}
					_ = outBufs[0].ToFlatData(outData)
					_ = outBufs[0].Finalize()
				}

				b.ResetTimer()
				b.SetBytes(int64(size*4*4 + 4)) // 4 input float32 arrays + 1 output scalar float32
				for i := 0; i < b.N; i++ {
					inA, _ := backend.BufferFromFlatData(0, dataSlice, shape)
					inB, _ := backend.BufferFromFlatData(0, dataSlice, shape)
					inC, _ := backend.BufferFromFlatData(0, dataSlice, shape)
					inD, _ := backend.BufferFromFlatData(0, dataSlice, shape)
					outBufs, errExec := exec.Execute([]compute.Buffer{inA, inB, inC, inD}, []bool{true, true, true, true}, 0)
					if errExec != nil {
						b.Fatalf("Execute failed: %+v", errExec)
					}
					errRead := outBufs[0].ToFlatData(outData)
					if errRead != nil {
						b.Fatalf("ToFlatData failed: %+v", errRead)
					}
					_ = outBufs[0].Finalize()
				}
				b.StopTimer()
			})
		}
	})
}
