CREATE TABLE runs (
  run_id        uuid PRIMARY KEY,
  root_run_id   uuid NOT NULL,
  parent_run_id uuid REFERENCES runs(run_id),
  tenant_id     text NOT NULL,
  principal     jsonb NOT NULL,
  spec_id       text NOT NULL,
  spec_version  int NOT NULL,
  request_class text NOT NULL CHECK (request_class IN ('interactive','background')),
  status        text NOT NULL CHECK (status IN ('running','waiting','wrapping_up',
                                                'succeeded','failed','cancelled')),
  depth         int NOT NULL DEFAULT 0,
  goal_hash     text,
  budget        jsonb NOT NULL,
  result        jsonb,
  created_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX runs_root ON runs(root_run_id);

CREATE UNIQUE INDEX runs_goal_dedup ON runs(root_run_id, goal_hash)
  WHERE goal_hash IS NOT NULL AND status IN ('running','succeeded');

CREATE TABLE run_events (
  event_id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  run_id          uuid NOT NULL REFERENCES runs(run_id),
  seq             int NOT NULL,
  node_id         text NOT NULL,
  kind            text NOT NULL,
  payload         jsonb NOT NULL,
  idempotency_key text NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  UNIQUE (run_id, idempotency_key),
  UNIQUE (run_id, seq)
);

CREATE INDEX run_events_run_seq ON run_events(run_id, seq);

CREATE TABLE node_tasks (
  task_id          uuid PRIMARY KEY,
  run_id           uuid NOT NULL REFERENCES runs(run_id),
  node_id          text NOT NULL,
  state            text NOT NULL CHECK (state IN ('pending','claimed','waiting','done','failed')),
  attempt          int NOT NULL DEFAULT 0,
  priority         int NOT NULL DEFAULT 0,
  available_at     timestamptz NOT NULL DEFAULT now(),
  lease_owner      text,
  lease_expires_at timestamptz,
  input            jsonb,
  created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX node_tasks_claim ON node_tasks (available_at, priority) WHERE state = 'pending';

