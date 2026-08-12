-- Migration 000097: Embedding dimensions configurable (1536 → 2048 for NVIDIA NIM)
-- WARNING: This migration NULL-ifies all existing embeddings and rebuilds indexes.
-- The gateway must be STOPPED before running this migration to prevent race conditions
-- where backfill goroutines write stale-dimension vectors during the DDL window.
-- After migration, restart the gateway — the backfill goroutines will automatically
-- re-embed all content from source texts.

BEGIN;

-- 1. Null out all embeddings (source texts preserved)
UPDATE memory_chunks SET embedding = NULL;
UPDATE skills SET embedding = NULL;
UPDATE agents SET embedding = NULL;
UPDATE team_tasks SET embedding = NULL;
UPDATE kg_entities SET embedding = NULL;
UPDATE episodic_summaries SET embedding = NULL;
UPDATE vault_documents SET embedding = NULL;

-- 2. Clear embedding cache (old vectors incompatible with new model)
DELETE FROM embedding_cache;

-- 3. Drop the vector_cosine_ops HNSW indexes BEFORE widening the columns.
--    Order is load-bearing: ALTER TYPE rebuilds every dependent index, and an
--    hnsw index over `vector` rejects anything above 2000 dimensions, so an
--    ALTER performed first aborts the whole migration.
DROP INDEX IF EXISTS idx_mem_vec;
DROP INDEX IF EXISTS idx_skills_embedding;
DROP INDEX IF EXISTS idx_agents_embedding;
DROP INDEX IF EXISTS idx_tt_embedding;
DROP INDEX IF EXISTS idx_kg_entity_vec;
DROP INDEX IF EXISTS idx_episodic_vec;
DROP INDEX IF EXISTS idx_episodic_embedding_hnsw;
DROP INDEX IF EXISTS idx_vault_docs_embedding;

-- 4. Widen the columns
ALTER TABLE memory_chunks ALTER COLUMN embedding TYPE vector(2048);
ALTER TABLE embedding_cache ALTER COLUMN embedding TYPE vector(2048);
ALTER TABLE skills ALTER COLUMN embedding TYPE vector(2048);
ALTER TABLE agents ALTER COLUMN embedding TYPE vector(2048);
ALTER TABLE team_tasks ALTER COLUMN embedding TYPE vector(2048);
ALTER TABLE kg_entities ALTER COLUMN embedding TYPE vector(2048);
ALTER TABLE episodic_summaries ALTER COLUMN embedding TYPE vector(2048);
ALTER TABLE vault_documents ALTER COLUMN embedding TYPE vector(2048);

-- 5. Recreate the indexes over a halfvec expression: hnsw over `vector` caps at
--    2000 dimensions, over `halfvec` at 4000. Predicates and storage parameters
--    are carried over verbatim from the pre-migration definitions.
--    Note: embedding_cache has no HNSW index — nothing to recreate.
CREATE INDEX idx_mem_vec ON memory_chunks USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops);

CREATE INDEX idx_skills_embedding ON skills USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops);

CREATE INDEX idx_agents_embedding ON agents USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops);

CREATE INDEX idx_tt_embedding ON team_tasks USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops);

CREATE INDEX idx_kg_entity_vec ON kg_entities USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops);

CREATE INDEX idx_episodic_vec ON episodic_summaries
    USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WHERE embedding IS NOT NULL;

CREATE INDEX idx_episodic_embedding_hnsw ON episodic_summaries
    USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE embedding IS NOT NULL;

CREATE INDEX idx_vault_docs_embedding ON vault_documents
    USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

COMMIT;
