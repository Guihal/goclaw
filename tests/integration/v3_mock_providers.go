//go:build integration

package integration

import (
	"context"
	"hash/fnv"
	"sync"

	"github.com/nextlevelbuilder/goclaw/internal/store"
)

// mockEmbedProvider returns deterministic vectors based on text content hash.
// Implements store.EmbeddingProvider for integration tests.
type mockEmbedProvider struct {
	dim int
	// lastInputType records the input_type of the most recent Embed call so
	// tests can assert query/passage classification. Guarded: stores fan
	// embedding calls out to goroutines (PGKnowledgeGraphStore.UpsertEntity),
	// so an unsynchronised field trips the race detector.
	mu            sync.Mutex
	lastInputType store.EmbeddingInputType
}

// newMockEmbedProvider emits vectors matching the migrated column width, so
// tests keep working when that width moves.
func newMockEmbedProvider() *mockEmbedProvider {
	return &mockEmbedProvider{dim: store.RequiredMemoryEmbeddingDimensions}
}

// newMockEmbedProviderDims builds a provider emitting vectors of the given
// width, for exercising a non-default embedding dimension end to end.
func newMockEmbedProviderDims(dim int) *mockEmbedProvider {
	return &mockEmbedProvider{dim: dim}
}

func (m *mockEmbedProvider) Name() string  { return "mock" }
func (m *mockEmbedProvider) Model() string { return "mock-embed-v1" }

// LastInputType returns the input_type of the most recent Embed call.
func (m *mockEmbedProvider) LastInputType() store.EmbeddingInputType {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastInputType
}

func (m *mockEmbedProvider) Embed(_ context.Context, texts []string, inputType store.EmbeddingInputType) ([][]float32, error) {
	m.mu.Lock()
	m.lastInputType = inputType
	m.mu.Unlock()
	result := make([][]float32, len(texts))
	for i, text := range texts {
		vec := make([]float32, m.dim)
		h := fnv.New32a()
		h.Write([]byte(text))
		seed := h.Sum32()
		for j := range vec {
			seed = seed*1103515245 + 12345
			vec[j] = float32(seed%1000) / 1000.0
		}
		result[i] = vec
	}
	return result, nil
}

// Compile-time interface check.
var _ store.EmbeddingProvider = (*mockEmbedProvider)(nil)
