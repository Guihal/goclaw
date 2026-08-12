//go:build integration

package integration

import (
	"context"
	"database/sql"
	"strconv"
	"strings"
	"testing"

	"github.com/nextlevelbuilder/goclaw/internal/store"
	"github.com/nextlevelbuilder/goclaw/internal/store/pg"
)

// embeddingDimsUnderTest pins the width migration 000097 gives the embedding
// columns. It is spelled out rather than read from
// store.RequiredMemoryEmbeddingDimensions on purpose: the point of the test is
// that the store templates the configured width into its SQL, so a constant
// that silently follows the code default would assert nothing.
const embeddingDimsUnderTest = 2048

// TestStoreMemory_ConfiguredDimsRoundTrip drives the write path (INSERT with a
// ::vector(N) cast), the cache path (batch upsert built with Fprintf) and the
// read path (ORDER BY over the ::halfvec(N) expression index). A width leaking
// through as a hardcoded literal fails here with a pgvector "expected N
// dimensions" error rather than silently degrading.
func TestStoreMemory_ConfiguredDimsRoundTrip(t *testing.T) {
	db := testDB(t)
	tenantID, agentID := seedTenantAgent(t, db)
	ctx := tenantCtx(tenantID)

	cfg := pg.DefaultPGMemoryConfig()
	cfg.EmbeddingDims = embeddingDimsUnderTest
	ms := pg.NewPGMemoryStore(db, cfg)
	provider := newMockEmbedProviderDims(embeddingDimsUnderTest)
	ms.SetEmbeddingProvider(provider)

	aid := agentID.String()
	uid := "user-dims-" + agentID.String()[:8]
	const path = "notes/dims.md"

	if err := ms.PutDocument(ctx, aid, uid, path, "pgvector stores halfvec embeddings for semantic recall"); err != nil {
		t.Fatalf("PutDocument: %v", err)
	}
	if err := ms.IndexDocument(ctx, aid, uid, path); err != nil {
		t.Fatalf("IndexDocument: %v", err)
	}
	if provider.LastInputType() != store.EmbedInputPassage {
		t.Errorf("indexing input_type = %q, want %q", provider.LastInputType(), store.EmbedInputPassage)
	}

	var stored int
	if err := db.QueryRow(
		`SELECT count(*) FROM memory_chunks WHERE tenant_id = $1 AND embedding IS NOT NULL`,
		tenantID,
	).Scan(&stored); err != nil {
		t.Fatalf("count chunks: %v", err)
	}
	if stored == 0 {
		t.Fatal("no chunk stored with an embedding")
	}

	results, err := ms.Search(ctx, "semantic recall", aid, uid, store.MemorySearchOptions{MaxResults: 5})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("search returned no results")
	}
	if provider.LastInputType() != store.EmbedInputQuery {
		t.Errorf("search input_type = %q, want %q", provider.LastInputType(), store.EmbedInputQuery)
	}

	// The cache write runs inside indexing; purging must clear it so a later
	// identity change cannot replay wrong-width vectors.
	purged, err := ms.PurgeEmbeddingCache(ctx)
	if err != nil {
		t.Fatalf("PurgeEmbeddingCache: %v", err)
	}
	if purged < 0 {
		t.Fatalf("purged = %d, want >= 0", purged)
	}
	var remaining int
	if err := db.QueryRow(`SELECT count(*) FROM embedding_cache`).Scan(&remaining); err != nil {
		t.Fatalf("count cache: %v", err)
	}
	if remaining != 0 {
		t.Errorf("embedding_cache rows after purge = %d, want 0", remaining)
	}

	assertVectorSearchUsesIndex(t, db)

	// Same store, a width the column cannot hold. Without this the test could
	// not tell a templated width from the compiled-in default, since the two
	// happen to be equal. pgvector rejects the whole row, so the chunk count
	// stays where it was.
	narrowCfg := pg.DefaultPGMemoryConfig()
	narrowCfg.EmbeddingDims = embeddingDimsUnderTest / 2
	narrow := pg.NewPGMemoryStore(db, narrowCfg)
	narrow.SetEmbeddingProvider(newMockEmbedProviderDims(narrowCfg.EmbeddingDims))

	const narrowPath = "notes/narrow.md"
	if err := narrow.PutDocument(ctx, aid, uid, narrowPath, "a document embedded at the wrong width"); err != nil {
		t.Fatalf("PutDocument narrow: %v", err)
	}
	if err := narrow.IndexDocument(ctx, aid, uid, narrowPath); err != nil {
		t.Fatalf("IndexDocument narrow: %v", err)
	}
	var narrowChunks int
	if err := db.QueryRow(
		`SELECT count(*) FROM memory_chunks WHERE tenant_id = $1 AND path = $2 AND embedding IS NOT NULL`,
		tenantID, narrowPath,
	).Scan(&narrowChunks); err != nil {
		t.Fatalf("count narrow chunks: %v", err)
	}
	if narrowChunks != 0 {
		t.Errorf("chunks written at width %d = %d, want 0 — the configured width is not reaching the SQL",
			narrowCfg.EmbeddingDims, narrowChunks)
	}
}

// TestStoreMemory_InvalidateAndBackfillRoundTrip covers what happens when the
// embedding model is swapped without a migration: the vectors already stored
// came from another model, so they must be dropped and rebuilt rather than left
// to be compared against new ones. The rebuild is the half that matters — an
// invalidation that nobody undoes is just data loss.
func TestStoreMemory_InvalidateAndBackfillRoundTrip(t *testing.T) {
	db := testDB(t)
	tenantID, agentID := seedTenantAgent(t, db)
	ctx := tenantCtx(tenantID)

	cfg := pg.DefaultPGMemoryConfig()
	cfg.EmbeddingDims = embeddingDimsUnderTest
	ms := pg.NewPGMemoryStore(db, cfg)
	ms.SetEmbeddingProvider(newMockEmbedProviderDims(embeddingDimsUnderTest))

	aid := agentID.String()
	uid := "user-inv-" + agentID.String()[:8]
	const path = "notes/invalidate.md"

	if err := ms.PutDocument(ctx, aid, uid, path, "a document that outlives the model that embedded it"); err != nil {
		t.Fatalf("PutDocument: %v", err)
	}
	if err := ms.IndexDocument(ctx, aid, uid, path); err != nil {
		t.Fatalf("IndexDocument: %v", err)
	}

	countEmbedded := func() int {
		t.Helper()
		var n int
		if err := db.QueryRow(
			`SELECT count(*) FROM memory_chunks WHERE tenant_id = $1 AND embedding IS NOT NULL`,
			tenantID,
		).Scan(&n); err != nil {
			t.Fatalf("count embedded chunks: %v", err)
		}
		return n
	}

	embedded := countEmbedded()
	if embedded == 0 {
		t.Fatal("no embedded chunk to invalidate")
	}

	dropped, err := ms.InvalidateEmbeddings(ctx)
	if err != nil {
		t.Fatalf("InvalidateEmbeddings: %v", err)
	}
	if dropped < int64(embedded) {
		t.Errorf("dropped = %d, want at least the %d embedded chunks", dropped, embedded)
	}
	if left := countEmbedded(); left != 0 {
		t.Fatalf("chunks still carrying a vector after invalidation = %d, want 0", left)
	}

	// Source text survives, so the backfill can rebuild what was dropped.
	var chunkText string
	if err := db.QueryRow(
		`SELECT text FROM memory_chunks WHERE tenant_id = $1 LIMIT 1`, tenantID,
	).Scan(&chunkText); err != nil {
		t.Fatalf("read chunk text: %v", err)
	}
	if chunkText == "" {
		t.Fatal("invalidation destroyed the source text — nothing could be re-embedded")
	}

	if _, err := ms.BackfillEmbeddings(ctx); err != nil {
		t.Fatalf("BackfillEmbeddings: %v", err)
	}
	if got := countEmbedded(); got != embedded {
		t.Fatalf("re-embedded chunks = %d, want %d", got, embedded)
	}
}

// assertVectorSearchUsesIndex checks that the cast form the store templates is
// the one the hnsw expression index was built on. A cast that stops matching
// still returns correct rows, just via a full sort, so nothing else in the
// suite would notice. It asserts the cast↔index match, not the exact plan of
// any production query: real searches add filters that change the shape.
//
// Sequential scans are disabled for the check: the fixture holds a handful of
// rows, where a seq scan is genuinely cheaper and says nothing about whether
// the index could serve the query. The setting is session-scoped, hence the
// dedicated connection.
func assertVectorSearchUsesIndex(t *testing.T, db *sql.DB) {
	t.Helper()

	values := make([]string, embeddingDimsUnderTest)
	for i := range values {
		values[i] = "0.01"
	}
	probe := "[" + strings.Join(values, ",") + "]"

	ctx := context.Background()
	conn, err := db.Conn(ctx)
	if err != nil {
		t.Fatalf("acquire conn: %v", err)
	}
	defer conn.Close()

	if _, err := conn.ExecContext(ctx, "SET enable_seqscan = off"); err != nil {
		t.Fatalf("disable seqscan: %v", err)
	}

	rows, err := conn.QueryContext(ctx, `
		EXPLAIN SELECT id FROM memory_chunks
		ORDER BY embedding::halfvec(`+strconv.Itoa(embeddingDimsUnderTest)+`) <=> $1::halfvec(`+strconv.Itoa(embeddingDimsUnderTest)+`)
		LIMIT 5`, probe)
	if err != nil {
		t.Fatalf("explain vector search: %v", err)
	}
	defer rows.Close()

	var plan strings.Builder
	for rows.Next() {
		var line string
		if err := rows.Scan(&line); err != nil {
			t.Fatalf("scan plan line: %v", err)
		}
		plan.WriteString(line)
		plan.WriteByte('\n')
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("iterate plan: %v", err)
	}
	if !strings.Contains(plan.String(), "Index Scan using idx_mem_vec") {
		t.Errorf("vector search plan does not use idx_mem_vec:\n%s", plan.String())
	}
}
