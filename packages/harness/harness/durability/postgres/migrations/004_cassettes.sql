CREATE TABLE cassettes (
  cassette_type text NOT NULL CHECK (cassette_type IN ('llm','tool')),
  cache_key     text NOT NULL,
  payload       jsonb NOT NULL,
  created_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (cassette_type, cache_key)
);
