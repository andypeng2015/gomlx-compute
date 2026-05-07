// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nontransposed

import (
	"fmt"
	"testing"
)

func TestPackLHS(t *testing.T) {
	lhsL1KernelRows := 4
	
	testCases := []struct {
		rows, cols int
		rowStart, colStart int
	}{
		{5, 20, 0, 0},
		{5, 20, 2, 3},
		{4, 16, 0, 0},
		{4, 15, 0, 0},
		{8, 32, 0, 0},
		{3, 10, 0, 0},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%dx%d_at_%d_%d", tc.rows, tc.cols, tc.rowStart, tc.colStart), func(t *testing.T) {
			totalRows := tc.rows + tc.rowStart + 2
			totalCols := tc.cols + tc.colStart + 2
			src := make([]float32, totalRows*totalCols)
			for i := range src {
				src[i] = float32(i + 1)
			}

			// Calculate dst size
			numStrips := (tc.rows + lhsL1KernelRows - 1) / lhsL1KernelRows
			dstSize := numStrips * tc.cols * lhsL1KernelRows
			dstNoSIMD := make([]float32, dstSize)
			dstAVX512 := make([]float32, dstSize)

			packLHS(src, dstNoSIMD, tc.rowStart, tc.colStart, totalCols, tc.rows, tc.cols, lhsL1KernelRows)
			avx512PackLHSFloat32(src, dstAVX512, tc.rowStart, tc.colStart, totalCols, tc.rows, tc.cols, lhsL1KernelRows)

			for i := range dstNoSIMD {
				if dstNoSIMD[i] != dstAVX512[i] {
					t.Errorf("Mismatch at index %d: noSIMD=%f, avx512=%f", i, dstNoSIMD[i], dstAVX512[i])
					if i > 10 { break }
				}
			}
		})
	}
}

func TestPackRHS(t *testing.T) {
	contractingRows := 3
	rhsCols := 5
	rhsL1KernelCols := 4
	src := make([]float32, contractingRows*rhsCols)
	for i := range src {
		src[i] = float32(i + 1)
	}

	// Calculate dst size: ceil(rhsCols/rhsL1KernelCols) * contractingRows * rhsL1KernelCols
	numStrips := (rhsCols + rhsL1KernelCols - 1) / rhsL1KernelCols
	dstSize := numStrips * contractingRows * rhsL1KernelCols
	dst := make([]float32, dstSize)

	packRHS(src, dst, 0, 0, rhsCols, contractingRows, rhsCols, rhsL1KernelCols)

	// We don't have an AVX512 version for packRHS yet, so we just verify the noSIMD one.
	// Expected layout: strips of width rhsL1KernelCols.
	// Strip 0: cols 0, 1, 2, 3. Strip 1: col 4 (padded).
	// Strip 0, Row 0: [1, 2, 3, 4]
	// Strip 0, Row 1: [6, 7, 8, 9]
	// Strip 0, Row 2: [11, 12, 13, 14]
	// Strip 1, Row 0: [5, 0, 0, 0]
	// Strip 1, Row 1: [10, 0, 0, 0]
	// Strip 1, Row 2: [15, 0, 0, 0]
	
	expected := []float32{
		1, 2, 3, 4,
		6, 7, 8, 9,
		11, 12, 13, 14,
		5, 0, 0, 0,
		10, 0, 0, 0,
		15, 0, 0, 0,
	}

	for i := range expected {
		if dst[i] != expected[i] {
			t.Errorf("Mismatch at index %d: got %f, expected %f", i, dst[i], expected[i])
		}
	}
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
