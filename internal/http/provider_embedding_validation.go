package http

import (
	"fmt"

	"github.com/nextlevelbuilder/goclaw/internal/store"
)

// Provider-level embedding settings are used by the memory system, whose
// PostgreSQL schema stores vector(N) embeddings where N comes from MemoryConfig.EmbeddingDimensions.
func validateProviderEmbeddingSettings(p *store.LLMProviderData, resolvedDims int) error {
	es := store.ParseEmbeddingSettings(p.Settings)
	if es == nil || !es.Enabled {
		return nil
	}
	if es.Dimensions < 0 {
		return fmt.Errorf("embedding.dimensions must be a positive integer or omitted")
	}
	// Same fallback as handleVerifyEmbedding: an unset resolvedDims would
	// otherwise compare against 0 and reject every explicit width.
	if resolvedDims <= 0 {
		resolvedDims = store.RequiredMemoryEmbeddingDimensions
	}
	if es.Dimensions > 0 && es.Dimensions != resolvedDims {
		return fmt.Errorf(
			"embedding.dimensions must be %d (configured) or omitted, got %d",
			resolvedDims,
			es.Dimensions,
		)
	}
	return nil
}
