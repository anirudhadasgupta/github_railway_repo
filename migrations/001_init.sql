-- Enable useful extensions for keyword search (optional but recommended)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS repos (
  id              BIGSERIAL PRIMARY KEY,
  owner           TEXT NOT NULL,
  name            TEXT NOT NULL,
  default_branch  TEXT,
  indexed_at      TIMESTAMPTZ,
  UNIQUE(owner, name)
);

CREATE TABLE IF NOT EXISTS files (
  id              BIGSERIAL PRIMARY KEY,
  repo_id         BIGINT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
  path            TEXT NOT NULL,
  ref             TEXT NOT NULL,
  blob_sha        TEXT,
  size_bytes      BIGINT,
  updated_at      TIMESTAMPTZ DEFAULT now(),
  UNIQUE(repo_id, path, ref)
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id        TEXT PRIMARY KEY,
  repo_id         BIGINT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
  repo_name       TEXT NOT NULL,
  ref             TEXT NOT NULL,
  path            TEXT NOT NULL,
  language        TEXT,
  symbol          TEXT,
  start_line      INT,
  end_line        INT,
  content_hash    TEXT NOT NULL,
  text            TEXT NOT NULL,
  created_at      TIMESTAMPTZ DEFAULT now()
);

-- Fast keyword search (optional; useful once chunk table grows)
CREATE INDEX IF NOT EXISTS chunks_text_trgm_idx ON chunks USING gin (text gin_trgm_ops);

CREATE TABLE IF NOT EXISTS index_jobs (
  job_id          TEXT PRIMARY KEY,
  owner           TEXT NOT NULL,
  repo            TEXT NOT NULL,
  ref             TEXT NOT NULL,
  status          TEXT NOT NULL, -- PENDING|RUNNING|DONE|ERROR
  progress        REAL NOT NULL DEFAULT 0,
  message         TEXT,
  created_at      TIMESTAMPTZ DEFAULT now(),
  updated_at      TIMESTAMPTZ DEFAULT now()
);
