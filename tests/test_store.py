"""Saved plans: content-derived ids, the in-memory store and the Supabase REST calls (against a stand-in)."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from antarctic_routing.store import SUMMARY, PlanStore, plan_id, summary


def make_plan(issue: str = "2023-11-14", upper: float = 0.0311) -> dict:
    return {"status": "recommended", "metadata": {"mode": "historical", "issue_date": issue},
            "locations": {"origin": {"id": "a", "name": "Alpha Point"}, "destination": {"id": "b", "name": None}},
            "route": {"departure_date": issue, "expected_hours": 30.25, "distance_km": 123.4},
            "risk": {"combined": {"p_breach_upper": upper}}}


REQUEST = {"origin": {"preset": "a"}, "destination": {"preset": "b"}, "issue": "2023-11-14", "window_days": 14}


def test_the_id_comes_from_the_content_and_the_summary_from_the_plan():
    plan = make_plan()
    assert plan_id(plan) == plan_id(json.loads(json.dumps(plan)))            # key order and identity do not matter
    assert plan_id(plan) != plan_id(make_plan(upper=0.04))
    row = summary("x", plan, "2026-01-01T00:00:00+00:00")
    assert tuple(row) == SUMMARY
    assert row == {"id": "x", "created_at": "2026-01-01T00:00:00+00:00", "origin": "Alpha Point", "destination": "b",
                   "issue_date": "2023-11-14", "mode": "historical", "status": "recommended",
                   "departure_date": "2023-11-14", "p_breach_upper": 0.0311, "expected_hours": 30.25,
                   "distance_km": 123.4}
    assert summary("y", {"status": "no_route"})["origin"] is None            # a plan without a route still lists


def test_memory_store_keeps_the_newest_plans_once_each():
    store = PlanStore(keep=2)
    assert store.storage == "memory"
    ids = [store.save(REQUEST, make_plan(f"2023-11-{d}")) for d in (14, 15, 14, 16)]
    assert ids[0] == ids[2] and len(set(ids)) == 3
    assert [r["issue_date"] for r in store.list()] == ["2023-11-16", "2023-11-14"]      # newest first, no duplicate
    assert store.get(ids[1]) is None                                                    # the oldest was dropped
    assert store.get(ids[3]) == {"request": REQUEST, "plan": make_plan("2023-11-16")}
    assert len(store.list(limit=1)) == 1


def test_serve_takes_only_supabase_values_from_a_local_dotenv(tmp_path, monkeypatch):
    from antarctic_routing.cli import _supabase_from_dotenv

    for name in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "ANTROUTE_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "from-the-shell")
    env = tmp_path / ".env"
    env.write_text("# comment\nANTROUTE_CONFIG=/app/config/config.yaml\nSUPABASE_URL = 'https://x.supabase.co'\n"
                   "SUPABASE_SERVICE_KEY=from-the-file\nSUPABASE_EMPTY=\n", encoding="utf-8")
    _supabase_from_dotenv(env)
    assert os.environ["SUPABASE_URL"] == "https://x.supabase.co"
    assert os.environ["SUPABASE_SERVICE_KEY"] == "from-the-shell"            # the environment wins
    assert "ANTROUTE_CONFIG" not in os.environ and "SUPABASE_EMPTY" not in os.environ
    _supabase_from_dotenv(tmp_path / "missing.env")                          # no file: nothing happens


def test_half_a_supabase_configuration_stays_in_memory():
    assert PlanStore("https://x.supabase.co", None).storage == "memory"
    assert PlanStore(None, "sb_secret_x").storage == "memory"


class FakePostgrest(BaseHTTPRequestHandler):
    """The two PostgREST calls the store makes, on a list of rows."""

    rows: list[dict] = []
    seen: list[dict] = []

    def log_message(self, *args):
        pass

    def _reply(self, status: int, body=None):
        raw = b"" if body is None else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _note(self) -> dict:
        url = urlsplit(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        self.seen.append({"method": self.command, "path": url.path, "query": q, "apikey": self.headers.get("apikey"),
                          "auth": self.headers.get("Authorization"), "prefer": self.headers.get("Prefer")})
        return q

    def do_POST(self):
        self._note()
        row = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not any(r["id"] == row["id"] for r in self.rows):                 # resolution=ignore-duplicates
            self.rows.append({"created_at": f"2026-01-01T00:00:0{len(self.rows)}+00:00", **row})
        self._reply(201)

    def do_GET(self):
        q = self._note()
        cols = q["select"].split(",")
        rows = [r for r in self.rows if "id" not in q or q["id"] == f"eq.{r['id']}"]
        if q.get("order") == "created_at.desc":
            rows = sorted(rows, key=lambda r: r["created_at"], reverse=True)
        self._reply(200, [{c: r[c] for c in cols} for r in rows[:int(q.get("limit", 1000))]])


@pytest.fixture()
def supabase():
    FakePostgrest.rows, FakePostgrest.seen = [], []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakePostgrest)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_supabase_store_saves_lists_and_reads_through_the_rest_api(supabase):
    store = PlanStore(supabase + "/", "sb_secret_test")
    assert store.storage == "supabase"
    first, second = store.save(REQUEST, make_plan()), store.save(REQUEST, make_plan("2023-11-15"))
    assert store.save(REQUEST, make_plan()) == first
    assert [r["id"] for r in FakePostgrest.rows] == [first, second]          # the repeat was not stored twice
    stored = FakePostgrest.rows[0]
    assert "created_at" in stored and stored["plan"] == make_plan() and stored["request"] == REQUEST
    assert [r["id"] for r in store.list()] == [second, first]
    assert tuple(store.list()[0]) == SUMMARY

    fresh = PlanStore(supabase, "sb_secret_test")                            # another process: nothing in memory
    assert fresh.get(first) == {"request": REQUEST, "plan": make_plan()}
    assert fresh.get("00000000-0000-4000-8000-000000000000") is None

    post = FakePostgrest.seen[0]
    assert (post["method"], post["path"], post["query"]) == ("POST", "/rest/v1/plans", {"on_conflict": "id"})
    assert post["prefer"] == "resolution=ignore-duplicates,return=minimal"
    assert {s["apikey"] for s in FakePostgrest.seen} == {"sb_secret_test"}
    assert {s["auth"] for s in FakePostgrest.seen} == {None}                 # a secret key is never a bearer token


def test_a_legacy_service_role_token_is_also_sent_as_the_bearer(supabase):
    PlanStore(supabase, "eyJhbGciOi.test").list()
    assert FakePostgrest.seen[0]["auth"] == "Bearer eyJhbGciOi.test"


def test_an_unreachable_supabase_never_fails_a_plan(caplog):
    store = PlanStore("http://127.0.0.1:9", "sb_secret_test")                # nothing listens on the discard port
    pid = store.save(REQUEST, make_plan())
    assert [r["id"] for r in store.list()] == [pid]                          # served from memory instead
    assert store.get(pid)["plan"] == make_plan()
    assert store.get("00000000-0000-4000-8000-000000000000") is None
    assert "Supabase save failed" in caplog.text and "sb_secret_test" not in caplog.text
