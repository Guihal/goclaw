package http

import (
	"encoding/json"
	"testing"

	"github.com/nextlevelbuilder/goclaw/internal/store"
)

func embeddingProvider(dims int) *store.LLMProviderData {
	settings := map[string]any{
		"embedding": map[string]any{
			"enabled":    true,
			"model":      "test-model",
			"dimensions": dims,
		},
	}
	raw, _ := json.Marshal(settings)
	return &store.LLMProviderData{Settings: raw}
}

func TestValidateProviderEmbeddingSettings(t *testing.T) {
	tests := []struct {
		name         string
		p            *store.LLMProviderData
		resolvedDims int
		wantErr      bool
	}{
		{"nil settings default", &store.LLMProviderData{}, 1536, false},
		{"embedding disabled", &store.LLMProviderData{
			Settings: json.RawMessage(`{"embedding":{"enabled":false}}`),
		}, 1536, false},
		{"dimensions omitted (0)", &store.LLMProviderData{
			Settings: json.RawMessage(`{"embedding":{"enabled":true,"model":"m"}}`),
		}, 1536, false},
		{"dimensions 1536 match default", embeddingProvider(1536), 1536, false},
		{"dimensions 2048 mismatch default", embeddingProvider(2048), 1536, true},
		{"dimensions 2048 match configured", embeddingProvider(2048), 2048, false},
		{"dimensions 1536 mismatch 2048 configured", embeddingProvider(1536), 2048, true},
		{"dimensions 1024 mismatch", embeddingProvider(1024), 1536, true},
		{"dimensions negative", embeddingProvider(-1), 1536, true},
		// resolvedDims unset must fall back to the schema width instead of
		// comparing against 0 and rejecting every explicit value.
		{"unset resolvedDims accepts schema width",
			embeddingProvider(store.RequiredMemoryEmbeddingDimensions), 0, false},
		{"unset resolvedDims rejects other width",
			embeddingProvider(store.RequiredMemoryEmbeddingDimensions / 2), 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateProviderEmbeddingSettings(tt.p, tt.resolvedDims)
			if (err != nil) != tt.wantErr {
				t.Fatalf("validateProviderEmbeddingSettings() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
