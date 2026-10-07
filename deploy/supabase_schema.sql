-- Saved route plans (see src/antarctic_routing/store.py and docs/DEPLOYMENT.md).
-- Run once in the Supabase project's SQL editor. Safe to run again.

create table if not exists public.plans (
  id              uuid primary key,                    -- derived from the plan's content by the API
  created_at      timestamptz not null default now(),
  origin          text,
  destination     text,
  issue_date      date,
  mode            text,                                -- historical | forecast
  status          text,                                -- recommended | no_feasible_departure | no_route
  departure_date  date,
  p_breach_upper  double precision,
  expected_hours  double precision,
  distance_km     double precision,
  request         jsonb not null,                      -- the POST /real/plan request
  plan            jsonb not null                       -- its full response
);

create index if not exists plans_created_at_idx on public.plans (created_at desc);

-- Only the API server reads and writes this table, with the project's secret (service role) key.
-- Row level security is on with no policies, so the public (publishable / anon) key can do nothing.
alter table public.plans enable row level security;
revoke all on table public.plans from anon, authenticated;
grant select, insert on table public.plans to service_role;
