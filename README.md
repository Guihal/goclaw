<p align="center">
  <img src="_statics/goclaw-logo.svg" alt="GoClaw" height="160" />
</p>

<p align="center"><strong>GoClaw — fork with configurable embedding dimensions</strong></p>

<p align="center">
  <img src="https://img.shields.io/badge/fork_of-nextlevelbuilder%2Fgoclaw-blue?style=flat-square" alt="Fork" />
  <img src="https://img.shields.io/badge/Go_1.26-00ADD8?style=flat-square&logo=go&logoColor=white" alt="Go" />
  <img src="https://img.shields.io/badge/PostgreSQL_18_+_pgvector-316192?style=flat-square&logo=postgresql&logoColor=white" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey?style=flat-square" alt="License: CC BY-NC 4.0" />
</p>

---

> **This is a fork of [nextlevelbuilder/goclaw](https://github.com/nextlevelbuilder/goclaw).**
> All credit for GoClaw itself goes upstream. This fork exists for one reason: to run
> embedding models wider than 1536 dimensions. Everything else is upstream code.
>
> The full upstream README — feature list, architecture, desktop edition, channels — is
> kept verbatim in **[README.upstream.md](README.upstream.md)**.

## Why this fork exists

Upstream hardcodes the embedding width at 1536 across the schema, the SQL and the UI.
That rules out every model with a different output width — including
NVIDIA `nemotron-3-embed-1b`, which this fork was built to run at **2048 dimensions**.

Changing a constant is not enough: pgvector stores the width in the column type, HNSW
caps `vector` indexes at 2000 dimensions, and vectors produced by two different models
are not comparable even when the width matches. This fork addresses all three.

## What changed

**Width comes from config, not from a literal.**
`agents.defaults.memory.embedding_dimensions` is templated into every INSERT, UPDATE and
search query. Leave it unset and the schema width is used.

**Migration `000097` widens the columns to 2048.**
All eight embedding tables move together. HNSW indexes are dropped first (a `vector`
index cannot exceed 2000 dimensions) and rebuilt over a `halfvec` cast expression, which
raises the ceiling to 4000 while keeping full float32 precision in storage — the cast
lives in the index and the `ORDER BY`, never on the write path. The down migration
restores the original definitions byte for byte.

**Embedding identity with automatic invalidation.**
The tuple `provider + model + dimensions + api_base` is recorded in `system_configs`.
When it changes, the embedding cache is purged and previously stored vectors are dropped
so the existing backfill workers re-embed them from source text. Without this, a model
swap leaves vectors that still sort — and return confident nonsense. `api_base` is part
of the identity because the same model name served from two endpoints is two models.

**Startup width gate.**
If the configured width disagrees with the actual column type, the gateway logs the
mismatch and refuses to enable embeddings, instead of failing every INSERT at runtime
with semantic search quietly returning nothing.

**API and UI follow the config.**
`POST /v1/providers/{id}/verify-embedding` now returns `required_dimensions`; the web UI
renders the server's value instead of a hardcoded `1536`, across all five locales.

## Quick start

Requires Docker and a PostgreSQL 18 image with pgvector (pulled automatically).

```bash
git clone https://github.com/Guihal/goclaw.git
cd goclaw
./prepare-env.sh                 # generates .env with gateway token + encryption key
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d
curl http://localhost:18790/health     # → {"status":"ok","protocol":3}
```

Dashboard: <http://localhost:18790>.

To run at a width other than 2048, set it in `config.json` **and** ship a migration that
retypes the columns — the startup gate will refuse to run if the two disagree:

```json5
{ agents: { defaults: { memory: { embedding_dimensions: 2048 } } } }
```

## Verifying the fork's behaviour

```bash
go build ./... && go vet ./... && go test ./internal/... ./cmd/...

# Integration tests need a pgvector instance:
docker run -d --name pgtest -p 5433:5432 \
  -e POSTGRES_PASSWORD=test -e POSTGRES_DB=goclaw_test pgvector/pgvector:pg18
TEST_DATABASE_URL="postgres://postgres:test@localhost:5433/goclaw_test?sslmode=disable" \
  go test -race -tags integration ./tests/integration/
```

Two integration tests cover this fork specifically:
`TestStoreMemory_ConfiguredDimsRoundTrip` (the configured width really reaches the SQL,
and searches still hit the HNSW index) and `TestStoreMemory_InvalidateAndBackfillRoundTrip`
(an identity change drops stale vectors and the backfill rebuilds them).

## Known limits

- Invalidation runs **at gateway startup**. Change the embedding model through the UI and
  the stale vectors survive until the next restart.
- Re-embedding costs real API calls. An identity change on a large corpus is not free.
- The width is process-wide, not per tenant.

## Staying current with upstream

```bash
git remote add upstream https://github.com/nextlevelbuilder/goclaw.git
git fetch upstream && git merge upstream/dev
```

Conflicts to expect live in `internal/store/pg/*.go` (the width is templated into SQL that
upstream writes as literals) and in `migrations/` if upstream adds a migration numbered 97.

## License

Upstream's license applies unchanged: **Creative Commons Attribution-NonCommercial 4.0
International** (CC BY-NC 4.0). Copyright © 2025-2026 GoClaw Contributors. Non-commercial
use only; see [LICENSE](LICENSE). Changes in this fork are published under the same terms.
