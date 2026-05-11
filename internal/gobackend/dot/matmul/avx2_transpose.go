// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package matmul

import "simd/archsimd"

// avx2Transpose4x8x32bits transposes 4 rows of 8x32-bit elements (uint32) into
// 8 rows of 4 32-bit elements, where each 2 rows of 4 elements is output together in one YMM register.
func avx2Transpose4x8x32bits(v0, v1, v2, v3 archsimd.Uint32x8) (row0to2, row2to4, row4to6, row6to8 archsimd.Uint32x8) {
	var (
		t0Indices = [8]uint32{
			0, 8, 0, 0, // Strip 0
			1, 9, 0, 0, // Strip 1
		}
		t1Indices = [8]uint32{
			2, 10, 0, 0, // Strip 2
			3, 11, 0, 0, // Strip 3
		}
		t2Indices = [8]uint32{
			4, 12, 0, 0, // Strip 4
			5, 13, 0, 0, // Strip 5
		}
		t3Indices = [8]uint32{
			6, 14, 0, 0, // Strip 6
			7, 15, 0, 0, // Strip 7
		}
	)
	{
		t00 := v0.ConcatPermute(v2, archsimd.LoadUint32x8(&t0Indices))
		t01 := v1.ConcatPermute(v3, archsimd.LoadUint32x8(&t0Indices))
		row0to2 = t00.InterleaveLoGrouped(t01)
	}
	{
		t10 := v0.ConcatPermute(v2, archsimd.LoadUint32x8(&t1Indices))
		t11 := v1.ConcatPermute(v3, archsimd.LoadUint32x8(&t1Indices))
		row2to4 = t10.InterleaveLoGrouped(t11)
	}
	{
		t20 := v0.ConcatPermute(v2, archsimd.LoadUint32x8(&t2Indices))
		t21 := v1.ConcatPermute(v3, archsimd.LoadUint32x8(&t2Indices))
		row4to6 = t20.InterleaveLoGrouped(t21)
	}
	{
		t30 := v0.ConcatPermute(v2, archsimd.LoadUint32x8(&t3Indices))
		t31 := v1.ConcatPermute(v3, archsimd.LoadUint32x8(&t3Indices))
		row6to8 = t30.InterleaveLoGrouped(t31)
	}
	return
}

// avx2Transpose4x16x16bits transposes 4 rows of 16x16-bit elements (uint16) into
// 16 rows of 4 16-bit elements, where each 4 rows of 4 elements is output together in one YMM register.
func avx2Transpose4x16x16bits(v0, v1, v2, v3 archsimd.Uint16x16) (row0to4, row4to8, row8to12, row12to16 archsimd.Uint16x16) {
	var (
		// Columns 0-3 → rows 0-3
		t0Indices = [16]uint16{
			0, 16, 1, 17, 0, 0, 0, 0, // Lane 0: cols 0,1
			2, 18, 3, 19, 0, 0, 0, 0, // Lane 1: cols 2,3
		}
		// Columns 4-7 → rows 4-7
		t1Indices = [16]uint16{
			4, 20, 5, 21, 0, 0, 0, 0, // Lane 0: cols 4,5
			6, 22, 7, 23, 0, 0, 0, 0, // Lane 1: cols 6,7
		}
		// Columns 8-11 → rows 8-11
		t2Indices = [16]uint16{
			8, 24, 9, 25, 0, 0, 0, 0, // Lane 0: cols 8,9
			10, 26, 11, 27, 0, 0, 0, 0, // Lane 1: cols 10,11
		}
		// Columns 12-15 → rows 12-15
		t3Indices = [16]uint16{
			12, 28, 13, 29, 0, 0, 0, 0, // Lane 0: cols 12,13
			14, 30, 15, 31, 0, 0, 0, 0, // Lane 1: cols 14,15
		}
	)
	{
		t00 := v0.ConcatPermute(v2, archsimd.LoadUint16x16(&t0Indices))
		t01 := v1.ConcatPermute(v3, archsimd.LoadUint16x16(&t0Indices))
		row0to4 = t00.InterleaveLoGrouped(t01)
	}
	{
		t10 := v0.ConcatPermute(v2, archsimd.LoadUint16x16(&t1Indices))
		t11 := v1.ConcatPermute(v3, archsimd.LoadUint16x16(&t1Indices))
		row4to8 = t10.InterleaveLoGrouped(t11)
	}
	{
		t20 := v0.ConcatPermute(v2, archsimd.LoadUint16x16(&t2Indices))
		t21 := v1.ConcatPermute(v3, archsimd.LoadUint16x16(&t2Indices))
		row8to12 = t20.InterleaveLoGrouped(t21)
	}
	{
		t30 := v0.ConcatPermute(v2, archsimd.LoadUint16x16(&t3Indices))
		t31 := v1.ConcatPermute(v3, archsimd.LoadUint16x16(&t3Indices))
		row12to16 = t30.InterleaveLoGrouped(t31)
	}
	return
}

// avx2Transpose4x4x64bits transposes 4 rows of 4x64-bit elements (uint64) into
// 4 rows of 4 64-bit elements, where each 1 row of 4 elements is output together in one YMM register.
func avx2Transpose4x4x64bits(v0, v1, v2, v3 archsimd.Uint64x4) (row0, row1, row2, row3 archsimd.Uint64x4) {
	// We use Uint32x8 for Permute since it's more flexible for AVX2.
	u0 := v0.AsUint32x8()
	u1 := v1.AsUint32x8()
	u2 := v2.AsUint32x8()
	u3 := v3.AsUint32x8()

	// Indices to pick a 64-bit value (pair of 32-bit values) from ConcatPermute(A, B).
	// A is indices 0..7, B is 8..15.
	// We want to pick col C from each vector: [u0[C], u1[C], u2[C], u3[C]]
	// colIndices_C: picks [u0[C], u1[C], 0, 0, 0, 0, 0, 0] (actually it fills the rest with something, we'll combine)

	// combineIndices picks the first two 64-bit values from two vectors and combines them.
	combineIndices := [8]uint32{0, 1, 2, 3, 8, 9, 10, 11}
	idx := archsimd.LoadUint32x8(&combineIndices)

	// row0: [v0[0], v1[0], v2[0], v3[0]]
	indices0 := [8]uint32{0, 1, 8, 9, 0, 0, 0, 0}
	t0_01 := u0.ConcatPermute(u1, archsimd.LoadUint32x8(&indices0))
	t0_23 := u2.ConcatPermute(u3, archsimd.LoadUint32x8(&indices0))
	row0 = t0_01.ConcatPermute(t0_23, idx).AsUint64x4()

	// row1: [v0[1], v1[1], v2[1], v3[1]]
	indices1 := [8]uint32{2, 3, 10, 11, 0, 0, 0, 0}
	t1_01 := u0.ConcatPermute(u1, archsimd.LoadUint32x8(&indices1))
	t1_23 := u2.ConcatPermute(u3, archsimd.LoadUint32x8(&indices1))
	row1 = t1_01.ConcatPermute(t1_23, idx).AsUint64x4()

	// row2: [v0[2], v1[2], v2[2], v3[2]]
	indices2 := [8]uint32{4, 5, 12, 13, 0, 0, 0, 0}
	t2_01 := u0.ConcatPermute(u1, archsimd.LoadUint32x8(&indices2))
	t2_23 := u2.ConcatPermute(u3, archsimd.LoadUint32x8(&indices2))
	row2 = t2_01.ConcatPermute(t2_23, idx).AsUint64x4()

	// row3: [v0[3], v1[3], v2[3], v3[3]]
	indices3 := [8]uint32{6, 7, 14, 15, 0, 0, 0, 0}
	t3_01 := u0.ConcatPermute(u1, archsimd.LoadUint32x8(&indices3))
	t3_23 := u2.ConcatPermute(u3, archsimd.LoadUint32x8(&indices3))
	row3 = t3_01.ConcatPermute(t3_23, idx).AsUint64x4()

	return
}
