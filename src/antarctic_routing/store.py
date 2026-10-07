"""Saved route plans.

Every ``POST /real/plan`` result is kept under an id derived from its content, so the same plan is stored
once. The last few are always held in this process's memory; with ``SUPABASE_URL`` and
``SUPABASE_SERVICE_KEY`` set they are also written to the ``plans`` table of a Supabase project
(``deploy/supabase_schema.sql``) through its REST API, and survive restarts.

The key is a server-side secret: it is read here, sent only to the Supabase project, and never returned by
the API or written into the dashboard. A Supabase failure is logged and the plan is still answered.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import UTC, datetime

log = logging.getLogger("antarctic_routing.store")

TABLE = "plans"
SUMMARY = ("id", "created_at", "origin", "destination", "issue_date", "mode", "status", "departure_date",
           "p_breach_upper", "expected_hours", "distance_km")
TIMEOUT_S = 5


def plan_id(plan: dict) -> str:
    """A UUID from the plan's content: the same result always gets the same id."""
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return str(uuid.UUID(hex=digest[:32]))


def summary(pid: str, plan: dict, created_at: str | None = None) -> dict:
    """The listing row of a plan (also the table's plain columns)."""
    loc, route, meta = plan.get("locations") or {}, plan.get("route") or {}, plan.get("metadata") or {}
    combined = (plan.get("risk") or {}).get("combined") or {}

    def name(end: str) -> str | None:
        x = loc.get(end) or {}
        return x.get("name") or x.get("id")

    return {"id": pid, "created_at": created_at or datetime.now(UTC).isoformat(timespec="seconds"),
            "origin": name("origin"), "destination": name("destination"), "issue_date": meta.get("issue_date"),
            "mode": meta.get("mode"), "status": plan.get("status"), "departure_date": route.get("departure_date"),
            "p_breach_upper": combined.get("p_breach_upper"), "expected_hours": route.get("expected_hours"),
            "distance_km": route.get("distance_km")}


class PlanStore:
    def __init__(self, url: str | None = None, key: str | None = None, keep: int = 32) -> None:
        self.url, self.key, self.keep = (url or "").rstrip("/"), key or "", keep
        self.recent: dict[str, dict] = {}       # id -> {"summary", "request", "plan"}, oldest first
        self.lock = threading.Lock()
        if bool(self.url) != bool(self.key):
            log.warning("saved plans stay in memory: set both SUPABASE_URL and SUPABASE_SERVICE_KEY")
            self.url = self.key = ""

    @classmethod
    def from_env(cls) -> PlanStore:
        return cls(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY"))

    @property
    def storage(self) -> str:
        return "supabase" if self.url else "memory"

    def _call(self, method: str, query: str, body: dict | None = None, prefer: str | None = None):
        # A secret key goes in `apikey` only; a legacy service_role JWT is also the bearer token.
        headers = {"apikey": self.key, "Content-Type": "application/json", "User-Agent": "antarctic-routing"}
        if self.key.startswith("eyJ"):
            headers["Authorization"] = f"Bearer {self.key}"
        if prefer:
            headers["Prefer"] = prefer
        req = urllib.request.Request(f"{self.url}/rest/v1/{TABLE}?{query}", method=method, headers=headers,
                                     data=None if body is None else json.dumps(body).encode())
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as res:  # noqa: S310 - the configured project URL
            raw = res.read()
        return json.loads(raw) if raw else None

    def _remote(self, what: str, *args, **kwargs):
        """One Supabase call; a failure is logged (never raised) and answers ``None``."""
        try:
            return self._call(*args, **kwargs)
        except urllib.error.HTTPError as exc:
            log.error("Supabase %s failed: HTTP %s %s", what, exc.code, exc.read()[:300].decode("utf-8", "replace"))
        except (OSError, ValueError) as exc:
            log.error("Supabase %s failed: %s", what, exc)
        return None

    def save(self, request: dict, plan: dict) -> str:
        pid = plan_id(plan)
        row = summary(pid, plan)
        with self.lock:
            self.recent.pop(pid, None)
            self.recent[pid] = {"summary": row, "request": request, "plan": plan}
            while len(self.recent) > self.keep:
                self.recent.pop(next(iter(self.recent)))
        if self.url:
            stored = {k: v for k, v in row.items() if k != "created_at"} | {"request": request, "plan": plan}
            self._remote("save", "POST", "on_conflict=id", stored, "resolution=ignore-duplicates,return=minimal")
        return pid

    def list(self, limit: int = 20) -> list[dict]:
        if self.url:
            query = urllib.parse.urlencode({"select": ",".join(SUMMARY), "order": "created_at.desc", "limit": limit})
            rows = self._remote("list", "GET", query)
            if rows is not None:
                return rows
        with self.lock:
            return [x["summary"] for x in reversed(self.recent.values())][:limit]

    def get(self, pid: str) -> dict | None:
        """``{"request", "plan"}`` of a saved plan, or ``None``."""
        with self.lock:
            hit = self.recent.get(pid)
        if hit is not None:
            return {"request": hit["request"], "plan": hit["plan"]}
        if self.url:
            rows = self._remote("read", "GET", urllib.parse.urlencode({"id": f"eq.{pid}", "select": "request,plan"}))
            if rows:
                return rows[0]
        return None
