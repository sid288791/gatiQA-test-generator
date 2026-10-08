-- gatiQA test generator schema
-- Runs automatically on first postgres container start (docker-entrypoint-initdb.d).

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Evaluation profile: a named, versioned bundle of evaluator rules.
CREATE TABLE IF NOT EXISTS evaluation_profile (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name               VARCHAR(200) NOT NULL,
    description        VARCHAR(2000),
    default_threshold  NUMERIC(4,3) NOT NULL DEFAULT 0.800,
    created_at         TIMESTAMP NOT NULL DEFAULT now()
);

-- Rules composing an evaluation profile (one profile -> many rules).
CREATE TABLE IF NOT EXISTS evaluation_profile_rule (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id      UUID NOT NULL REFERENCES evaluation_profile(id) ON DELETE CASCADE,
    evaluator_type  VARCHAR(100) NOT NULL,
    weight          NUMERIC(5,2) NOT NULL DEFAULT 1.00,
    configuration   JSONB NOT NULL DEFAULT '{}'::jsonb,
    enabled         BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_epr_profile ON evaluation_profile_rule(profile_id);

-- Persisted test case (approved/generated spec stored as JSONB).
CREATE TABLE IF NOT EXISTS test_case (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                   VARCHAR(200) NOT NULL,
    description            VARCHAR(2000),
    test_spec              JSONB NOT NULL,
    evaluation_profile_id  UUID REFERENCES evaluation_profile(id) ON DELETE SET NULL,
    status                 VARCHAR(50) NOT NULL DEFAULT 'DRAFT',
    created_by             VARCHAR(200),
    created_at             TIMESTAMP NOT NULL DEFAULT now(),
    updated_at             TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_test_case_profile ON test_case(evaluation_profile_id);
CREATE INDEX IF NOT EXISTS idx_test_case_status  ON test_case(status);

-- Many-to-many: a test case can be linked to multiple evaluation profiles.
CREATE TABLE IF NOT EXISTS test_case_evaluation_profile (
    test_case_id  UUID NOT NULL REFERENCES test_case(id) ON DELETE CASCADE,
    profile_id    UUID NOT NULL REFERENCES evaluation_profile(id) ON DELETE CASCADE,
    created_at    TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (test_case_id, profile_id)
);

CREATE INDEX IF NOT EXISTS idx_tcep_profile ON test_case_evaluation_profile(profile_id);

-- Keep updated_at fresh on UPDATE.
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_test_case_updated ON test_case;
CREATE TRIGGER trg_test_case_updated
    BEFORE UPDATE ON test_case
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
