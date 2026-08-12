package cmd

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/nextlevelbuilder/goclaw/internal/config"
	"github.com/nextlevelbuilder/goclaw/internal/memory"
	"github.com/nextlevelbuilder/goclaw/internal/store"
)

// fakeSysConfigStore is an in-memory SystemConfigStore. Get reports a missing
// key as an error, matching PGSystemConfigStore.
type fakeSysConfigStore struct {
	values  map[string]string
	setErr  error
	setCall int
}

func newFakeSysConfigStore() *fakeSysConfigStore {
	return &fakeSysConfigStore{values: map[string]string{}}
}

func (f *fakeSysConfigStore) Get(_ context.Context, key string) (string, error) {
	v, ok := f.values[key]
	if !ok {
		return "", errors.New("system config not found: " + key)
	}
	return v, nil
}

func (f *fakeSysConfigStore) Set(_ context.Context, key, value string) error {
	f.setCall++
	if f.setErr != nil {
		return f.setErr
	}
	f.values[key] = value
	return nil
}

func (f *fakeSysConfigStore) Delete(_ context.Context, key string) error {
	delete(f.values, key)
	return nil
}

func (f *fakeSysConfigStore) List(_ context.Context) (map[string]string, error) {
	out := make(map[string]string, len(f.values))
	for k, v := range f.values {
		out[k] = v
	}
	return out, nil
}

type fakePurger struct {
	calls int
	err   error
}

func (p *fakePurger) PurgeEmbeddingCache(context.Context) (int64, error) {
	p.calls++
	if p.err != nil {
		return 0, p.err
	}
	return 3, nil
}

type fakeInvalidator struct {
	calls int
	err   error
}

func (i *fakeInvalidator) InvalidateEmbeddings(context.Context) (int64, error) {
	i.calls++
	if i.err != nil {
		return 0, i.err
	}
	return 7, nil
}

func TestReconcileEmbeddingIdentity(t *testing.T) {
	identity := embeddingIdentity{Provider: "nvidia", Model: "nemotron", Dimensions: 2048, APIBase: "https://integrate.api.nvidia.com/v1"}

	// seedIdentity records `id` on a fresh store and returns it ready for the
	// call under test, with the counters zeroed.
	seedIdentity := func(t *testing.T, id embeddingIdentity) (*fakeSysConfigStore, *fakePurger, *fakeInvalidator) {
		t.Helper()
		sc, purger, inv := newFakeSysConfigStore(), &fakePurger{}, &fakeInvalidator{}
		reconcileEmbeddingIdentity(context.Background(), sc, purger, inv, id)
		purger.calls, inv.calls, sc.setCall = 0, 0, 0
		return sc, purger, inv
	}

	t.Run("no record purges the cache but keeps vectors", func(t *testing.T) {
		sc, purger, inv := newFakeSysConfigStore(), &fakePurger{}, &fakeInvalidator{}
		reconcileEmbeddingIdentity(context.Background(), sc, purger, inv, identity)
		if purger.calls != 1 {
			t.Fatalf("purge calls = %d, want 1", purger.calls)
		}
		// A fresh install or a just-migrated database has no stale vectors, and
		// dropping them would throw away a corpus that cost real money to embed.
		if inv.calls != 0 {
			t.Errorf("invalidate calls = %d, want 0 — nothing was recorded to be stale against", inv.calls)
		}
		if got := sc.values[EmbeddingIdentityConfigKey]; got == "" {
			t.Fatal("identity not recorded")
		}
	})

	t.Run("unchanged identity is a no-op", func(t *testing.T) {
		sc, purger, inv := seedIdentity(t, identity)

		reconcileEmbeddingIdentity(context.Background(), sc, purger, inv, identity)
		if purger.calls != 0 || inv.calls != 0 {
			t.Errorf("purge calls = %d, invalidate calls = %d, want 0 and 0", purger.calls, inv.calls)
		}
		if sc.setCall != 0 {
			t.Errorf("set calls = %d, want 0", sc.setCall)
		}
	})

	t.Run("model change drops stale vectors", func(t *testing.T) {
		sc, purger, inv := seedIdentity(t, identity)

		swapped := identity
		swapped.Model = "text-embedding-3-small"
		reconcileEmbeddingIdentity(context.Background(), sc, purger, inv, swapped)
		if purger.calls != 1 {
			t.Errorf("purge calls = %d, want 1", purger.calls)
		}
		if inv.calls != 1 {
			t.Errorf("invalidate calls = %d, want 1 — vectors of the old model would stay searchable", inv.calls)
		}
	})

	t.Run("dimensions-only change purges and invalidates", func(t *testing.T) {
		sc, purger, inv := seedIdentity(t, identity)

		narrowed := identity
		narrowed.Dimensions = 1536
		reconcileEmbeddingIdentity(context.Background(), sc, purger, inv, narrowed)
		if purger.calls != 1 || inv.calls != 1 {
			t.Fatalf("purge calls = %d, invalidate calls = %d, want 1 and 1", purger.calls, inv.calls)
		}
	})

	t.Run("api_base change drops stale vectors", func(t *testing.T) {
		sc, purger, inv := seedIdentity(t, identity)

		// Same model name, different endpoint: different weights, incomparable
		// vectors, and nothing else in the tuple moves to signal it.
		rehomed := identity
		rehomed.APIBase = "http://localhost:8000/v1"
		reconcileEmbeddingIdentity(context.Background(), sc, purger, inv, rehomed)
		if inv.calls != 1 {
			t.Fatalf("invalidate calls = %d, want 1", inv.calls)
		}
	})

	t.Run("legacy record without api_base does not re-embed", func(t *testing.T) {
		sc, purger, inv := newFakeSysConfigStore(), &fakePurger{}, &fakeInvalidator{}
		legacy := identity
		legacy.APIBase = ""
		reconcileEmbeddingIdentity(context.Background(), sc, purger, inv, legacy)
		purger.calls, inv.calls, sc.setCall = 0, 0, 0

		// Upgrading to a build that records api_base must not look like an
		// endpoint change, or every deployment re-embeds its whole corpus once.
		reconcileEmbeddingIdentity(context.Background(), sc, purger, inv, identity)
		if purger.calls != 0 || inv.calls != 0 {
			t.Errorf("purge calls = %d, invalidate calls = %d, want 0 and 0", purger.calls, inv.calls)
		}
		if sc.setCall != 1 {
			t.Errorf("set calls = %d, want 1 — the record should be upgraded in place", sc.setCall)
		}
		if got := sc.values[EmbeddingIdentityConfigKey]; !strings.Contains(got, "integrate.api.nvidia.com") {
			t.Errorf("record not upgraded to the new shape: %s", got)
		}
	})

	t.Run("failed purge leaves the record stale", func(t *testing.T) {
		sc := newFakeSysConfigStore()
		purger := &fakePurger{err: errors.New("boom")}

		reconcileEmbeddingIdentity(context.Background(), sc, purger, &fakeInvalidator{}, identity)
		if _, ok := sc.values[EmbeddingIdentityConfigKey]; ok {
			t.Fatal("identity recorded despite failed purge — next start would skip the purge")
		}
	})

	t.Run("failed invalidation leaves the record stale", func(t *testing.T) {
		sc, purger, _ := seedIdentity(t, identity)
		before := sc.values[EmbeddingIdentityConfigKey]
		failing := &fakeInvalidator{err: errors.New("boom")}

		swapped := identity
		swapped.Model = "other"
		reconcileEmbeddingIdentity(context.Background(), sc, purger, failing, swapped)
		if got := sc.values[EmbeddingIdentityConfigKey]; got != before {
			t.Fatalf("identity advanced to %s despite failed invalidation — stale vectors would look current", got)
		}
	})

	t.Run("nil invalidator still purges", func(t *testing.T) {
		sc, purger, _ := seedIdentity(t, identity)

		swapped := identity
		swapped.Model = "other"
		reconcileEmbeddingIdentity(context.Background(), sc, purger, nil, swapped)
		if purger.calls != 1 {
			t.Errorf("purge calls = %d, want 1", purger.calls)
		}
		if sc.setCall != 1 {
			t.Errorf("set calls = %d, want 1", sc.setCall)
		}
	})

	t.Run("nil store is tolerated", func(t *testing.T) {
		reconcileEmbeddingIdentity(context.Background(), nil, &fakePurger{}, &fakeInvalidator{}, identity)
	})
}

// stubEmbedProvider is the minimum memory.EmbeddingProvider needed to build an
// identity; APIBase is added by embedding it in stubEmbedProviderWithBase.
type stubEmbedProvider struct{ name, model string }

func (s stubEmbedProvider) Name() string  { return s.name }
func (s stubEmbedProvider) Model() string { return s.model }
func (s stubEmbedProvider) Embed(context.Context, []string, memory.EmbeddingInputType) ([][]float32, error) {
	return nil, nil
}

type stubEmbedProviderWithBase struct {
	stubEmbedProvider
	base string
}

func (s stubEmbedProviderWithBase) APIBase() string { return s.base }

func TestNewEmbeddingIdentity(t *testing.T) {
	t.Run("reads api_base when the provider reports one", func(t *testing.T) {
		p := stubEmbedProviderWithBase{stubEmbedProvider{"nvidia", "nemotron"}, "https://integrate.api.nvidia.com/v1/"}
		got := newEmbeddingIdentity(p, 2048)
		want := embeddingIdentity{Provider: "nvidia", Model: "nemotron", Dimensions: 2048, APIBase: "https://integrate.api.nvidia.com/v1"}
		if got != want {
			t.Fatalf("newEmbeddingIdentity = %+v, want %+v", got, want)
		}
	})

	t.Run("provider without an api_base yields an empty one", func(t *testing.T) {
		got := newEmbeddingIdentity(stubEmbedProvider{"local", "bge"}, 1024)
		if got.APIBase != "" {
			t.Fatalf("APIBase = %q, want empty", got.APIBase)
		}
	})
}

func TestNormalizeAPIBaseIgnoresCosmeticDifferences(t *testing.T) {
	// Re-embedding a corpus costs money; a trailing slash must not trigger it.
	cases := []string{"https://api.x/v1", "https://api.x/v1/", " https://api.x/v1// "}
	for _, in := range cases {
		if got := normalizeAPIBase(in); got != "https://api.x/v1" {
			t.Errorf("normalizeAPIBase(%q) = %q, want %q", in, got, "https://api.x/v1")
		}
	}
}

type fakeColumnInspector struct {
	dims int
	err  error
}

func (f fakeColumnInspector) EmbeddingColumnDims(context.Context) (int, error) {
	return f.dims, f.err
}

func TestEmbeddingWidthMatchesSchema(t *testing.T) {
	cases := []struct {
		name       string
		inspector  embeddingColumnInspector
		configured int
		want       bool
	}{
		{"width agrees", fakeColumnInspector{dims: 2048}, 2048, true},
		{"width disagrees", fakeColumnInspector{dims: 1536}, 2048, false},
		{"untyped column passes", fakeColumnInspector{dims: 0}, 2048, true},
		{"unreadable column passes", fakeColumnInspector{err: errors.New("boom")}, 2048, true},
		{"no inspector passes", nil, 2048, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := embeddingWidthMatchesSchema(context.Background(), tc.inspector, tc.configured); got != tc.want {
				t.Fatalf("embeddingWidthMatchesSchema = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestResolvedEmbeddingDimsFallsBackToSchemaWidth(t *testing.T) {
	if got := resolvedEmbeddingDims(nil); got != store.RequiredMemoryEmbeddingDimensions {
		t.Fatalf("resolvedEmbeddingDims(nil) = %d, want %d", got, store.RequiredMemoryEmbeddingDimensions)
	}
}

func TestResolvedEmbeddingDimsHonorsConfig(t *testing.T) {
	cfg := &config.Config{}
	cfg.Agents.Defaults.Memory = &config.MemoryConfig{EmbeddingDimensions: 1024}
	if got := resolvedEmbeddingDims(cfg); got != 1024 {
		t.Fatalf("resolvedEmbeddingDims = %d, want 1024", got)
	}
}
