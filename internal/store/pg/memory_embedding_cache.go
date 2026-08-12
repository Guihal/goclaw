package pg

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"time"
)

// embeddingCacheEntry holds data for a single cache row.
type embeddingCacheEntry struct {
	Hash      string
	Embedding []float32
}

// lookupEmbeddingCache fetches cached embeddings for the given content hashes.
// Returns a map from hash -> embedding vector. Missing hashes are simply absent.
func (s *PGMemoryStore) lookupEmbeddingCache(ctx context.Context, hashes []string, provider, model string) (map[string][]float32, error) {
	if len(hashes) == 0 {
		return nil, nil
	}

	// Build positional params: $1..$N for hashes, $N+1 for provider, $N+2 for model
	placeholders := make([]string, len(hashes))
	args := make([]any, 0, len(hashes)+2)
	for i, h := range hashes {
		placeholders[i] = fmt.Sprintf("$%d", i+1)
		args = append(args, h)
	}
	args = append(args, provider, model)

	query := fmt.Sprintf(
		"SELECT hash, embedding FROM embedding_cache WHERE hash IN (%s) AND provider = $%d AND model = $%d",
		strings.Join(placeholders, ","), len(hashes)+1, len(hashes)+2,
	)

	type cacheRow struct {
		Hash      string `db:"hash"`
		Embedding string `db:"embedding"`
	}
	var cacheRows []cacheRow
	if err := pkgSqlxDB.SelectContext(ctx, &cacheRows, query, args...); err != nil {
		return nil, fmt.Errorf("lookup embedding cache: %w", err)
	}

	result := make(map[string][]float32, len(hashes))
	for _, row := range cacheRows {
		vec, err := parseVector(row.Embedding)
		if err != nil {
			slog.Warn("embedding cache parse error", "hash", row.Hash, "error", err)
			continue
		}
		result[row.Hash] = vec
	}
	return result, nil
}

// EmbeddingColumnDims reports the width the embedding column is actually typed
// with, read from the catalog rather than assumed. memory_chunks stands in for
// every embedding column: migration 000097 moves them as one unit.
// Returns 0 when the column carries no width (an untyped `vector`).
func (s *PGMemoryStore) EmbeddingColumnDims(ctx context.Context) (int, error) {
	var typmod int
	err := s.db.QueryRowContext(ctx, `
		SELECT a.atttypmod
		FROM pg_attribute a
		WHERE a.attrelid = 'memory_chunks'::regclass AND a.attname = 'embedding'`,
	).Scan(&typmod)
	if err != nil {
		return 0, fmt.Errorf("read embedding column width: %w", err)
	}
	if typmod < 0 {
		return 0, nil
	}
	return typmod, nil
}

// PurgeEmbeddingCache deletes every cached vector and returns the row count.
// Called when the embedding identity (provider/model/dimensions) changes:
// cache rows carry no provenance beyond (hash, provider, model), so vectors of
// a stale dimension would be replayed into a column typed for the new one.
// Intentionally cross-tenant — the identity is process-wide, not per tenant.
func (s *PGMemoryStore) PurgeEmbeddingCache(ctx context.Context) (int64, error) {
	res, err := s.db.ExecContext(ctx, "DELETE FROM embedding_cache")
	if err != nil {
		return 0, fmt.Errorf("purge embedding cache: %w", err)
	}
	return res.RowsAffected()
}

// embeddingTables lists every table holding a persisted vector produced by the
// embedding provider. Must stay in step with the ALTER TABLE list in migration
// 000097: a table that can hold a vector but is missing here keeps vectors of a
// retired model forever, which is worse than having none.
var embeddingTables = []string{
	"memory_chunks",
	"skills",
	"agents",
	"team_tasks",
	"kg_entities",
	"episodic_summaries",
	"vault_documents",
}

// InvalidateEmbeddings nulls every persisted vector and returns the row count.
//
// Called when the embedding identity changes without a migration: the stored
// vectors came from a different model, so a cosine distance against a new query
// vector is a meaningless number that still sorts, which is why they must go
// rather than be left to degrade quietly. Source text is untouched, so the
// backfill workers rebuild them. One transaction: a half-invalidated corpus
// mixes two models in one index, exactly the state this prevents.
// Intentionally cross-tenant — the identity is process-wide, not per tenant.
func (s *PGMemoryStore) InvalidateEmbeddings(ctx context.Context) (int64, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("invalidate embeddings: begin: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	var total int64
	for _, table := range embeddingTables {
		res, err := tx.ExecContext(ctx, "UPDATE "+table+" SET embedding = NULL WHERE embedding IS NOT NULL")
		if err != nil {
			return 0, fmt.Errorf("invalidate embeddings in %s: %w", table, err)
		}
		affected, err := res.RowsAffected()
		if err != nil {
			return 0, fmt.Errorf("invalidate embeddings in %s: rows affected: %w", table, err)
		}
		total += affected
	}

	if err := tx.Commit(); err != nil {
		return 0, fmt.Errorf("invalidate embeddings: commit: %w", err)
	}
	return total, nil
}

// writeEmbeddingCache batch-upserts embedding cache entries.
// Gracefully skips on dimension mismatch against the configured vector width.
func (s *PGMemoryStore) writeEmbeddingCache(ctx context.Context, entries []embeddingCacheEntry, provider, model string) error {
	if len(entries) == 0 {
		return nil
	}

	now := time.Now()
	tenantID := tenantIDForInsert(ctx)
	dims := s.resolvedDims()

	// Process in batches of 100 to avoid exceeding max query params
	const batchSize = 100
	for start := 0; start < len(entries); start += batchSize {
		end := min(start+batchSize, len(entries))
		batch := entries[start:end]

		var sb strings.Builder
		sb.WriteString(`INSERT INTO embedding_cache (hash, provider, model, embedding, dims, created_at, updated_at, tenant_id) VALUES `)
		args := make([]any, 0, len(batch)*7)
		for i, e := range batch {
			if i > 0 {
				sb.WriteByte(',')
			}
			base := i * 7
			fmt.Fprintf(&sb, "($%d,$%d,$%d,$%d::vector(%d),$%d,$%d,$%d,$%d)",
				base+1, base+2, base+3, base+4, dims, base+5, base+6, base+6, base+7)
			args = append(args, e.Hash, provider, model, vectorToString(e.Embedding), len(e.Embedding), now, tenantID)
		}
		sb.WriteString(` ON CONFLICT (hash, provider, model) DO UPDATE SET embedding = EXCLUDED.embedding, dims = EXCLUDED.dims, updated_at = EXCLUDED.updated_at`)

		_, err := s.db.ExecContext(ctx, sb.String(), args...)
		if err != nil {
			// pgvector dimension mismatch — skip cache gracefully
			if strings.Contains(err.Error(), "dimensions") {
				slog.Warn("embedding cache skipped: vector dimension mismatch",
					"provider", provider, "model", model,
					"actual_dims", len(batch[0].Embedding), "error", err)
				return nil
			}
			return fmt.Errorf("batch write embedding cache: %w", err)
		}
	}
	return nil
}

// parseVector converts a pgvector string like "[0.1,0.2,0.3]" into []float32.
func parseVector(s string) ([]float32, error) {
	s = strings.TrimSpace(s)
	if len(s) < 2 {
		return nil, fmt.Errorf("vector string too short: %q", s)
	}
	// Strip surrounding brackets ([] from pgvector, () as fallback)
	s = strings.TrimPrefix(s, "[")
	s = strings.TrimSuffix(s, "]")
	s = strings.TrimPrefix(s, "(")
	s = strings.TrimSuffix(s, ")")
	if s == "" {
		return nil, nil
	}

	parts := strings.Split(s, ",")
	vec := make([]float32, 0, len(parts))
	for _, p := range parts {
		f, err := strconv.ParseFloat(strings.TrimSpace(p), 32)
		if err != nil {
			return nil, fmt.Errorf("parse vector element %q: %w", p, err)
		}
		vec = append(vec, float32(f))
	}
	return vec, nil
}
