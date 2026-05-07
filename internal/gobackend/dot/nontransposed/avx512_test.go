// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package nontransposed_test

import (
	"testing"

	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/dot/nontransposed"
	"github.com/gomlx/compute/support/backendtest"
)

func TestAVX512(t *testing.T) {
	// Force AVX512 variant only for NonTransposed.
	defer func() {
		dot.ResetTestRegistrations()
	}()
	dot.ResetTestRegistrations()
	nontransposed.RegisterAVX512ForTests()

	backendtest.TestDotGeneral(t, backend)
}
