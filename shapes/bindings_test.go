// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package shapes

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/stretchr/testify/require"
)

func TestBindings(t *testing.T) {
	t.Run("AxisBindings_Key", func(t *testing.T) {
		b := AxisBindings{"batch": 32, "seq_len": 128}
		require.Equal(t, "batch=32,seq_len=128", b.Key())

		// Deterministic ordering.
		b2 := AxisBindings{"seq_len": 128, "batch": 32}
		require.Equal(t, b.Key(), b2.Key())

		// Empty.
		require.Equal(t, "", AxisBindings{}.Key())
	})

	t.Run("Resolve", func(t *testing.T) {
		s := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
		bindings := AxisBindings{"batch": 32}
		resolved, err := s.Resolve(bindings)
		require.NoError(t, err)

		require.Equal(t, []int{32, 512}, resolved.Dimensions)
		require.Equal(t, []string{"batch", ""}, resolved.AxisNames)
		require.False(t, resolved.IsDynamic())
	})

	t.Run("Resolve_MultipleAxes", func(t *testing.T) {
		s := MakeDynamic(dtypes.Float32, []int{-1, -1, 768}, []string{"batch", "seq_len", ""})
		bindings := AxisBindings{"batch": 8, "seq_len": 256}
		resolved, err := s.Resolve(bindings)
		require.NoError(t, err)

		require.Equal(t, []int{8, 256, 768}, resolved.Dimensions)
	})

	t.Run("Resolve_StaticShape", func(t *testing.T) {
		s := Make(dtypes.Float32, 32, 512)
		// Resolve on static shape returns same shape (no-op).
		resolved, err := s.Resolve(AxisBindings{"batch": 64})
		require.NoError(t, err)
		require.True(t, s.Equal(resolved))
	})

	t.Run("Resolve_MissingBinding", func(t *testing.T) {
		s := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
		_, err := s.Resolve(AxisBindings{})
		require.Error(t, err)
	})

	t.Run("Resolve_NonPositiveBinding", func(t *testing.T) {
		s := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
		_, err := s.Resolve(AxisBindings{"batch": 0})
		require.Error(t, err)
		_, err = s.Resolve(AxisBindings{"batch": -5})
		require.Error(t, err)
	})

	t.Run("Extract", func(t *testing.T) {
		t.Run("Basic", func(t *testing.T) {
			bindings := make(AxisBindings)
			template := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
			concrete := Make(dtypes.Float32, 32, 512)
			err := bindings.Extract(template, concrete)
			require.NoError(t, err)
			require.Equal(t, AxisBindings{"batch": 32}, bindings)
		})

		t.Run("MultipleAxes", func(t *testing.T) {
			template := MakeDynamic(dtypes.Float32, []int{-1, -1, 768}, []string{"batch", "seq_len", ""})
			concrete := Make(dtypes.Float32, 8, 128, 768)
			bindings := make(AxisBindings)
			err := bindings.Extract(template, concrete)
			require.NoError(t, err)
			require.Equal(t, AxisBindings{"batch": 8, "seq_len": 128}, bindings)
		})

		t.Run("ConsistencyCheck", func(t *testing.T) {
			// Same axis name appears multiple times with different values.
			template := MakeDynamic(dtypes.Float32, []int{-1, -1}, []string{"n", "n"})
			concrete := Make(dtypes.Float32, 5, 5)

			// Same value → OK.
			bindings := make(AxisBindings)
			err := bindings.Extract(template, concrete)
			require.NoError(t, err)
			require.Equal(t, AxisBindings{"n": 5}, bindings)

			// Different values → error.
			concrete2 := Make(dtypes.Float32, 5, 10)
			bindings2 := make(AxisBindings)
			err = bindings2.Extract(template, concrete2)
			require.Error(t, err)
			require.Contains(t, err.Error(), "conflicting")
		})

		t.Run("RankMismatch", func(t *testing.T) {
			template := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
			concrete := Make(dtypes.Float32, 32, 512, 3)
			bindings := make(AxisBindings)
			err := bindings.Extract(template, concrete)
			require.Error(t, err)
			require.Contains(t, err.Error(), "rank")
		})

		t.Run("StaticDimMismatch", func(t *testing.T) {
			template := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
			concrete := Make(dtypes.Float32, 32, 256)
			b := make(AxisBindings)
			err := b.Extract(template, concrete)
			require.Error(t, err)
			require.Contains(t, err.Error(), "mismatch")
		})
	})

	t.Run("UnifyAxisName", func(t *testing.T) {
		// Both empty.
		name, err := UnifyAxisName("", "")
		require.NoError(t, err)
		require.Equal(t, "", name)

		// One named.
		name, err = UnifyAxisName("batch", "")
		require.NoError(t, err)
		require.Equal(t, "batch", name)

		name, err = UnifyAxisName("", "batch")
		require.NoError(t, err)
		require.Equal(t, "batch", name)

		// Same name.
		name, err = UnifyAxisName("batch", "batch")
		require.NoError(t, err)
		require.Equal(t, "batch", name)

		// Different names.
		_, err = UnifyAxisName("batch", "time")
		require.Error(t, err)
		require.Contains(t, err.Error(), "incompatible")
	})

	t.Run("UnifyAxisNames", func(t *testing.T) {
		s1 := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
		s2 := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})

		names, err := UnifyAxisNames(s1, s2)
		require.NoError(t, err)
		require.Equal(t, []string{"batch", ""}, names)

		// One unnamed adopts.
		s3 := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
		s4 := Make(dtypes.Float32, 32, 512) // no AxisNames
		names, err = UnifyAxisNames(s3, s4)
		require.NoError(t, err)
		require.Equal(t, []string{"batch", ""}, names)

		// Both nil.
		s5 := Make(dtypes.Float32, 32, 512)
		s6 := Make(dtypes.Float32, 32, 512)
		names, err = UnifyAxisNames(s5, s6)
		require.NoError(t, err)
		require.Nil(t, names)

		// Conflict.
		s7 := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
		s8 := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"time", ""})
		_, err = UnifyAxisNames(s7, s8)
		require.Error(t, err)
	})

	t.Run("RoundTrip_ExtractAndResolve", func(t *testing.T) {
		// Extract bindings from concrete shape, then resolve template with those bindings.
		template := MakeDynamic(dtypes.Float32, []int{-1, -1, 768}, []string{"batch", "seq_len", ""})
		concrete := Make(dtypes.Float32, 16, 64, 768)

		b := make(AxisBindings)
		err := b.Extract(template, concrete)
		require.NoError(t, err)

		resolved, err := template.Resolve(b)
		require.NoError(t, err)
		require.Equal(t, concrete.Dimensions, resolved.Dimensions)
		require.False(t, resolved.IsDynamic())
	})
}
