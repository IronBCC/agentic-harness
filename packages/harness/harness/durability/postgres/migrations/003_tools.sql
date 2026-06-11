CREATE TABLE tool_registry (
  tenant_id         text NOT NULL,
  tool_id           text NOT NULL,
  name              text NOT NULL,
  description       text NOT NULL,
  input_schema      jsonb NOT NULL,
  source            text NOT NULL,
  side_effect       text NOT NULL CHECK (side_effect IN ('pure','read','write')),
  idempotency       text NOT NULL CHECK (idempotency IN ('keyed','none')),
  freshness         text NOT NULL CHECK (freshness IN ('pure','session','volatile')),
  auth_mode         text NOT NULL CHECK (auth_mode IN ('service','user_passthrough')),
  requires_approval boolean NOT NULL DEFAULT false,
  index_card        text NOT NULL,
  metadata          jsonb NOT NULL DEFAULT '{}'::jsonb,
  updated_at        timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, tool_id)
);

CREATE TABLE memo_cache (
  tenant_id     text NOT NULL,
  tool_id       text NOT NULL,
  cache_key     text NOT NULL,
  output        jsonb NOT NULL,
  artifact_hint text,
  created_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, tool_id, cache_key)
);
