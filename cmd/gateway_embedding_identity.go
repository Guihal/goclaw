package cmd

import (
	"context"
	"encoding/json"
	"log/slog"
	"strings"

	"github.com/nextlevelbuilder/goclaw/internal/config"
	"github.com/nextlevelbuilder/goclaw/internal/memory"
	"github.com/nextlevelbuilder/goclaw/internal/store"
)

// resolvedEmbeddingDims is the single place the configured embedding width is
// derived, so store SQL, the HTTP validation gates and the recorded identity
// can never disagree about it.
func resolvedEmbeddingDims(cfg *config.Config) int {
	if cfg == nil || cfg.Agents.Defaults.Memory == nil {
		return store.ResolveEmbeddingDimensions(0)
	}
	return store.ResolveEmbeddingDimensions(cfg.Agents.Defaults.Memory.EmbeddingDimensions)
}

// EmbeddingIdentityConfigKey is the system_configs key holding the embedding
// identity the persisted vectors were produced with.
const EmbeddingIdentityConfigKey = "embedding.identity"

// embeddingIdentity is the tuple that makes a stored vector meaningful.
// Any field changing makes every persisted vector incomparable with new ones.
type embeddingIdentity struct {
	Provider   string `json:"provider"`
	Model      string `json:"model"`
	Dimensions int    `json:"dimensions"`
	// APIBase is part of the identity because model names are not globally
	// unique: "nemotron-3-embed-1b" served by NVIDIA and by some local
	// llama.cpp build produce different vectors under the same name.
	APIBase string `json:"api_base"`
}

// embeddingAPIBaseReporter is implemented by embedding providers that know the
// endpoint they call. Optional so that adding an endpoint-aware identity does
// not force every implementation to grow a method.
type embeddingAPIBaseReporter interface {
	APIBase() string
}

// newEmbeddingIdentity builds the identity of an embedding provider at the
// resolved width.
func newEmbeddingIdentity(p memory.EmbeddingProvider, dims int) embeddingIdentity {
	id := embeddingIdentity{Model: p.Model(), Dimensions: dims}
	id.Provider = p.Name()
	if reporter, ok := p.(embeddingAPIBaseReporter); ok {
		id.APIBase = normalizeAPIBase(reporter.APIBase())
	}
	return id
}

// normalizeAPIBase strips the differences that do not change which endpoint is
// called, so that cosmetic config edits do not trigger a re-embed of the whole
// corpus.
func normalizeAPIBase(base string) string {
	return strings.TrimRight(strings.TrimSpace(base), "/")
}

// matches reports whether two identities describe vectors that can be compared.
//
// A stored record written before api_base joined the identity has an empty
// APIBase. That is "unknown", not "different": treating it as a mismatch would
// make the first start after an upgrade re-embed every corpus in every
// deployment for no reason.
func (id embeddingIdentity) matches(stored embeddingIdentity) bool {
	if id.Provider != stored.Provider || id.Model != stored.Model || id.Dimensions != stored.Dimensions {
		return false
	}
	return stored.APIBase == "" || stored.APIBase == id.APIBase
}

// embeddingCachePurger is implemented by stores keeping an embedding cache.
// Optional capability: SQLite builds store no vectors and do not implement it.
type embeddingCachePurger interface {
	PurgeEmbeddingCache(ctx context.Context) (int64, error)
}

// embeddingVectorInvalidator is implemented by stores that can drop every
// persisted vector so the backfill workers re-embed from source text.
type embeddingVectorInvalidator interface {
	InvalidateEmbeddings(ctx context.Context) (int64, error)
}

// embeddingColumnInspector is implemented by stores that can report the width
// their embedding column is actually typed with.
type embeddingColumnInspector interface {
	EmbeddingColumnDims(ctx context.Context) (int, error)
}

// embeddingWiringDims returns the width every embedding write must use, or 0
// when embeddings must stay switched off because the configured width and the
// database column disagree. Resolved once at startup and handed to every
// subsystem that wires an embedding provider, so they cannot drift apart.
func embeddingWiringDims(pgStores *store.Stores, cfg *config.Config) int {
	dims := resolvedEmbeddingDims(cfg)
	if pgStores == nil || pgStores.Memory == nil {
		return dims
	}
	inspector, _ := pgStores.Memory.(embeddingColumnInspector)
	if !embeddingWidthMatchesSchema(context.Background(), inspector, dims) {
		return 0
	}
	return dims
}

// embeddingWidthMatchesSchema reports whether the configured embedding width
// agrees with the column it will be written to.
//
// The config knob and the column type must move together (config change +
// migration). Without this check a config-only edit produces no startup signal
// and instead fails every single INSERT at runtime, one warning per row, with
// semantic search quietly empty. Unknown widths (store cannot report, catalog
// read failed, untyped column) are treated as a match: refusing to run on a
// failed introspection would be worse than the mismatch it guards against.
func embeddingWidthMatchesSchema(ctx context.Context, inspector embeddingColumnInspector, configured int) bool {
	if inspector == nil {
		return true
	}
	actual, err := inspector.EmbeddingColumnDims(ctx)
	if err != nil {
		slog.Warn("embedding column width unreadable, skipping width check", "error", err)
		return true
	}
	if actual == 0 || actual == configured {
		return true
	}
	slog.Error("embedding width does not match the database column; embeddings disabled",
		"configured", configured, "column", actual,
		"fix", "set agents.defaults.memory.embedding_dimensions to the column width, or migrate the column to the configured width")
	return false
}

// reconcileEmbeddingIdentity records the active embedding identity in
// system_configs and drops the embedding cache whenever it does not match what
// was recorded before.
//
// Cache rows are keyed by (hash, provider, model) only, so a model that keeps
// its name while changing width — or a dimensions-only config change — would
// otherwise replay vectors of the wrong width into the halfvec column and fail
// every INSERT. An absent record is treated as a mismatch: the recorded
// identity is the only provenance available, and purging a cache is cheap
// against re-embedding with a silently wrong vector.
//
// Persisted vectors are dropped too when a PREVIOUS identity was recorded and
// differs from the current one: they were produced by another model and are not
// comparable with the ones written from now on, so leaving them makes semantic
// search return confident nonsense rather than fewer results. Dropping them is
// recoverable — the backfill goroutines re-embed from the source text, which is
// never deleted. No recorded identity means a fresh install or a database that
// migration 000097 already nulled, so there is nothing stale to drop.
func reconcileEmbeddingIdentity(
	ctx context.Context,
	sc store.SystemConfigStore,
	purger embeddingCachePurger,
	invalidator embeddingVectorInvalidator,
	current embeddingIdentity,
) {
	if sc == nil {
		return
	}
	next, err := json.Marshal(current)
	if err != nil {
		slog.Warn("embedding identity: marshal failed", "error", err)
		return
	}

	// Get errors on a missing key; both cases mean "no usable record".
	raw, getErr := sc.Get(ctx, EmbeddingIdentityConfigKey)
	var stored embeddingIdentity
	hasStored := getErr == nil && json.Unmarshal([]byte(raw), &stored) == nil
	if hasStored && current.matches(stored) {
		if raw == string(next) {
			return
		}
		// Same vectors, older record shape (an upgrade that added a field).
		// Rewrite it without touching any data.
		if err := sc.Set(ctx, EmbeddingIdentityConfigKey, string(next)); err != nil {
			slog.Warn("embedding identity: failed to record", "error", err)
		}
		return
	}

	if purger != nil {
		purged, purgeErr := purger.PurgeEmbeddingCache(ctx)
		if purgeErr != nil {
			// Leave the recorded identity stale so the next start retries the
			// purge instead of trusting a cache that was never cleared.
			slog.Warn("embedding identity changed but cache purge failed", "error", purgeErr)
			return
		}
		slog.Warn("embedding identity changed, embedding cache invalidated",
			"previous", raw, "current", string(next), "purged_rows", purged)
	}

	if hasStored && invalidator != nil {
		dropped, invErr := invalidator.InvalidateEmbeddings(ctx)
		if invErr != nil {
			// Same reasoning as the purge above: recording the new identity now
			// would strand vectors of the old model as if they were current.
			slog.Warn("embedding identity changed but stale vectors could not be dropped", "error", invErr)
			return
		}
		slog.Warn("embedding identity changed, stale vectors dropped and queued for re-embedding",
			"previous", raw, "current", string(next), "dropped_rows", dropped)
	}

	if err := sc.Set(ctx, EmbeddingIdentityConfigKey, string(next)); err != nil {
		slog.Warn("embedding identity: failed to record", "error", err)
	}
}
