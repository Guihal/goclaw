-- Rollback of 000097: embedding width back to 1536.
-- DESTRUCTIVE: 2048-dim vectors cannot be projected down to 1536, so every
-- re-embedded vector produced after the up-migration is lost. Source texts are
-- preserved and the backfill goroutines re-embed on the next gateway start,
-- which costs a full pass of embedding-API calls.
-- The gateway must be STOPPED before running this migration — a live backfill
-- goroutine writing during the DDL window would insert wrong-width vectors.
-- Also pin agents.defaults.memory.embedding_dimensions to 1536 in config before
-- restarting, otherwise the gateway writes vector(2048) values into a
-- vector(1536) column and every embedding INSERT fails.

BEGIN;

-- 1. Null out all embeddings (source texts preserved)
UPDATE memory_chunks SET embedding = NULL;
UPDATE skills SET embedding = NULL;
UPDATE agents SET embedding = NULL;
UPDATE team_tasks SET embedding = NULL;
UPDATE kg_entities SET embedding = NULL;
UPDATE episodic_summaries SET embedding = NULL;
UPDATE vault_documents SET embedding = NULL;

-- 2. Clear embedding cache (2048-dim vectors incompatible with vector(1536))
DELETE FROM embedding_cache;

-- 3. Drop the halfvec expression indexes before narrowing the columns, mirroring
--    the up-migration: ALTER TYPE rebuilds dependent indexes.
DROP INDEX IF EXISTS idx_mem_vec;
DROP INDEX IF EXISTS idx_skills_embedding;
DROP INDEX IF EXISTS idx_agents_embedding;
DROP INDEX IF EXISTS idx_tt_embedding;
DROP INDEX IF EXISTS idx_kg_entity_vec;
DROP INDEX IF EXISTS idx_episodic_vec;
DROP INDEX IF EXISTS idx_episodic_embedding_hnsw;
DROP INDEX IF EXISTS idx_vault_docs_embedding;

-- 4. Restore column types
ALTER TABLE memory_chunks ALTER COLUMN embedding TYPE vector(1536);
ALTER TABLE embedding_cache ALTER COLUMN embedding TYPE vector(1536);
ALTER TABLE skills ALTER COLUMN embedding TYPE vector(1536);
ALTER TABLE agents ALTER COLUMN embedding TYPE vector(1536);
ALTER TABLE team_tasks ALTER COLUMN embedding TYPE vector(1536);
ALTER TABLE kg_entities ALTER COLUMN embedding TYPE vector(1536);
ALTER TABLE episodic_summaries ALTER COLUMN embedding TYPE vector(1536);
ALTER TABLE vault_documents ALTER COLUMN embedding TYPE vector(1536);

-- 5. Restore the pre-000097 vector_cosine_ops indexes verbatim. Rolling the
--    migration back means rolling the binary back too, and pre-patch query SQL
--    compares plain `vector`, which a halfvec expression index cannot serve.
CREATE INDEX idx_mem_vec ON memory_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX idx_skills_embedding ON skills USING hnsw (embedding vector_cosine_ops);

CREATE INDEX idx_agents_embedding ON agents USING hnsw (embedding vector_cosine_ops);

CREATE INDEX idx_tt_embedding ON team_tasks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX idx_kg_entity_vec ON kg_entities USING hnsw (embedding vector_cosine_ops);

CREATE INDEX idx_episodic_vec ON episodic_summaries
    USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;

CREATE INDEX idx_episodic_embedding_hnsw ON episodic_summaries
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE embedding IS NOT NULL;

CREATE INDEX idx_vault_docs_embedding ON vault_documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 6. Drop the recorded embedding identity so the next start re-records it and
--    re-purges the cache instead of trusting a 2048-dim record.
DELETE FROM system_configs WHERE key = 'embedding.identity';

COMMIT;
