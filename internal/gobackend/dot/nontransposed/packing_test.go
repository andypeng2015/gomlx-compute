// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nontransposed

import (
	"fmt"
	"testing"
)

// PackLHSFn is the signature for LHS packing functions.
type PackLHSFn func(src, dst []float32, srcRowStart, srcColStart, srcRowStride, lhsRows, contractingCols, lhsL1KernelRows int)

// PackRHSFn is the signature for RHS packing functions.
type PackRHSFn func(src, dst []float32, srcRowStart, srcColStart, srcStrideCol, contractingRows, rhsCols, RHSL1KernelCols int)

func runPackLHSTests(t *testing.T, name string, packLHSFn PackLHSFn, lhsL1KernelRows int) {
	testCases := []struct {
		rows, cols         int
		rowStart, colStart int
	}{
		{rows: 5, cols: 20, rowStart: 0, colStart: 0},
		{rows: 5, cols: 20, rowStart: 2, colStart: 3},
		{rows: 4, cols: 16, rowStart: 0, colStart: 0},
		{rows: 4, cols: 15, rowStart: 0, colStart: 0},
		{rows: 8, cols: 32, rowStart: 0, colStart: 0},
		{rows: 3, cols: 10, rowStart: 0, colStart: 0},
		// Larger test cases
		{rows: 128, cols: 256, rowStart: 7, colStart: 11},
		{rows: 127, cols: 255, rowStart: 0, colStart: 0},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%s/LHS/%dx%d_at_%d_%d", name, tc.rows, tc.cols, tc.rowStart, tc.colStart), func(t *testing.T) {
			totalRows := tc.rows + tc.rowStart + 2
			totalCols := tc.cols + tc.colStart + 2
			src := make([]float32, totalRows*totalCols)
			for i := range src {
				src[i] = float32(i + 1)
			}

			numStrips := (tc.rows + lhsL1KernelRows - 1) / lhsL1KernelRows
			dstSize := numStrips * tc.cols * lhsL1KernelRows
			dstExpected := make([]float32, dstSize)
			dstActual := make([]float32, dstSize)

			// Reference implementation
			packLHS(src, dstExpected, tc.rowStart, tc.colStart, totalCols, tc.rows, tc.cols, lhsL1KernelRows)
			// Implementation under test
			packLHSFn(src, dstActual, tc.rowStart, tc.colStart, totalCols, tc.rows, tc.cols, lhsL1KernelRows)

			for i := range dstExpected {
				if dstExpected[i] != dstActual[i] {
					t.Fatalf("Mismatch at index %d: expected %f, got %f", i, dstExpected[i], dstActual[i])
				}
			}
		})
	}
}

func runPackRHSTests(t *testing.T, name string, packRHSFn PackRHSFn, rhsL1KernelCols int) {
	testCases := []struct {
		rows, cols         int
		rowStart, colStart int
	}{
		{rows: 3, cols: 32, rowStart: 0, colStart: 0},
		{rows: 5, cols: 64, rowStart: 2, colStart: 3},
		{rows: 10, cols: 100, rowStart: 0, colStart: 0},
		{rows: 3, cols: 10, rowStart: 0, colStart: 0},
		// Larger test cases
		{rows: 256, cols: 128, rowStart: 13, colStart: 17},
		{rows: 255, cols: 127, rowStart: 0, colStart: 0},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%s/RHS/%dx%d_at_%d_%d", name, tc.rows, tc.cols, tc.rowStart, tc.colStart), func(t *testing.T) {
			totalRows := tc.rows + tc.rowStart + 2
			totalCols := tc.cols + tc.colStart + 2
			src := make([]float32, totalRows*totalCols)
			for i := range src {
				src[i] = float32(i + 1)
			}

			numStrips := (tc.cols + rhsL1KernelCols - 1) / rhsL1KernelCols
			dstSize := numStrips * tc.rows * rhsL1KernelCols
			dstExpected := make([]float32, dstSize)
			dstActual := make([]float32, dstSize)

			// Reference implementation
			packRHS(src, dstExpected, tc.rowStart, tc.colStart, totalCols, tc.rows, tc.cols, rhsL1KernelCols)
			// Implementation under test
			packRHSFn(src, dstActual, tc.rowStart, tc.colStart, totalCols, tc.rows, tc.cols, rhsL1KernelCols)

			for i := range dstExpected {
				if dstExpected[i] != dstActual[i] {
					t.Fatalf("Mismatch at index %d: expected %f, got %f", i, dstExpected[i], dstActual[i])
				}
			}
		})
	}
}

func TestPackAVX512(t *testing.T) {
	runPackLHSTests(t, "AVX512", avx512PackLHSFloat32, 4)
	runPackRHSTests(t, "AVX512", avx512PackRHSFloat32, 32)
}

func TestApplyPackedOutput(t *testing.T) {
	height := 3
	width := 5
	packedOutputRowStride := 8
	outputRowStride := 6
	
	packedOutput := make([]float32, height*packedOutputRowStride)
	for i := range packedOutput {
		packedOutput[i] = float32(i + 1)
	}
	
	outputNoSIMD := make([]float32, height*outputRowStride + 2) // +2 for offset
	outputAVX512 := make([]float32, height*outputRowStride + 2)
	
	// Test isFirstContractingPanel = true
	noSIMDApplyPackedOutput(packedOutput, outputNoSIMD, true, packedOutputRowStride, 0, 1, outputRowStride, height, width)
	avx512ApplyPackedOutputFloat32(packedOutput, outputAVX512, true, packedOutputRowStride, 0, 1, outputRowStride, height, width)
	
	for i := range outputNoSIMD {
		if outputNoSIMD[i] != outputAVX512[i] {
			t.Errorf("First panel: Mismatch at index %d: noSIMD=%f, avx512=%f", i, outputNoSIMD[i], outputAVX512[i])
		}
	}
	
	// Test isFirstContractingPanel = false (accumulation)
	noSIMDApplyPackedOutput(packedOutput, outputNoSIMD, false, packedOutputRowStride, 0, 1, outputRowStride, height, width)
	avx512ApplyPackedOutputFloat32(packedOutput, outputAVX512, false, packedOutputRowStride, 0, 1, outputRowStride, height, width)

	for i := range outputNoSIMD {
		if outputNoSIMD[i] != outputAVX512[i] {
			t.Errorf("Accumulation: Mismatch at index %d: noSIMD=%f, avx512=%f", i, outputNoSIMD[i], outputAVX512[i])
		}
	}
}
