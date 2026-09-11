"""The log. This module is the schema from SCHEMA.md, expressed as code.

Two rules from SCHEMA.md are enforced here rather than trusted:

  - Only this module touches the database. Everyone else calls read_df().
  - Only columns in the contract can be written. Anything else raises.

If you need a column that isn't here, the conversation is with P3 and the
team, not with this file.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

EMBED_DIM = 32      # column 12 width
NUM_CLASSES = 10    # columns 13 and 14 width


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS requests (
    request_id        INTEGER PRIMARY KEY AUTOINCREMENT,  -- 1
    run_id            TEXT    NOT NULL,                   -- 2
    ts                REAL    NOT NULL,                   -- 3
    api_key_id        TEXT    NOT NULL,                   -- 4
    tier              TEXT    NOT NULL,                   -- 5
    ip                TEXT    NOT NULL,                   -- 6
    status_code       INTEGER NOT NULL,                   -- 7
    error_code        TEXT,                               -- 8
    latency_ms        REAL,                               -- 9
    input_sha256      TEXT,                               -- 10
    input_bytes       INTEGER,                            -- 11
    embedding         BLOB,                               -- 12
    model_probs       TEXT,                               -- 13
    returned_probs    TEXT,                               -- 14
    degradation_level INTEGER NOT NULL DEFAULT 0,         -- 15
    suspicion_score   REAL                                -- 16
);

CREATE INDEX IF NOT EXISTS ix_req_run_ts  ON requests (run_id, ts);
CREATE INDEX IF NOT EXISTS ix_req_key     ON requests (run_id, api_key_id, ts);
CREATE INDEX IF NOT EXISTS ix_req_ip      ON requests (run_id, ip, ts);

CREATE TABLE IF NOT EXISTS keys (
    api_key_id  TEXT PRIMARY KEY,
    secret      TEXT NOT NULL UNIQUE,
    tier        TEXT NOT NULL,
    owner       TEXT NOT NULL,
    created_ts  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_keys_secret ON keys (secret);
"""

# Columns 2-16. request_id is assigned by the database.
WRITABLE_COLS = [
    "run_id", "ts", "api_key_id", "tier", "ip", "status_code", "error_code",
    "latency_ms", "input_sha256", "input_bytes", "embedding", "model_probs",
    "returned_probs", "degradation_level", "suspicion_score",
]

_INSERT_SQL = (
    f"INSERT INTO requests ({', '.join(WRITABLE_COLS)}) "
    f"VALUES ({', '.join('?' * len(WRITABLE_COLS))})"
)


# ------------------------------------------------------------ column 12

def encode_embedding(vec) -> bytes | None:
    """32 floats -> 128 bytes, little-endian float32."""
    if vec is None:
        return None
    arr = np.asarray(vec, dtype="<f4").reshape(-1)
    if arr.size != EMBED_DIM:
        raise ValueError(f"embedding must be {EMBED_DIM}-d, got {arr.size}")
    return arr.tobytes()


def decode_embedding(blob) -> np.ndarray | None:
    """128 bytes -> numpy array of 32 float32. Nobody should call this by hand."""
    if not isinstance(blob, (bytes, bytearray)):
        return None
    return np.frombuffer(bytes(blob), dtype="<f4")


# ------------------------------------------------------- columns 13, 14

def encode_probs(probs) -> str | None:
    if probs is None:
        return None
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if p.size != NUM_CLASSES:
        raise ValueError(f"probs must be {NUM_CLASSES}-d, got {p.size}")
    return json.dumps([float(v) for v in p])


def decode_probs(s):
    if not isinstance(s, str) or not s:
        return None
    return json.loads(s)


class LogStore:
    """Thread-safe writer plus the one supported reader.

    One SQLite connection per thread, WAL mode. Comfortably handles a
    FastAPI worker at the volumes this project produces.
    """

    def __init__(self, path: str | Path, batch_size: int = 1):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._batch_size = max(1, batch_size)
        self._pending: list[tuple] = []
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    # ---------------------------------------------------- connection

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.row_factory = sqlite3.Row
            self._local.conn = c
        return c

    # ---------------------------------------------------- writing

    def log(self, **kw: Any) -> None:
        """Write one row. Columns outside the contract raise immediately.

        Failing loudly here is deliberate. A silently-dropped column is a
        detector that quietly trains on nothing, discovered on Day 3.
        """
        unknown = set(kw) - set(WRITABLE_COLS)
        if unknown:
            raise KeyError(
                f"not in SCHEMA.md: {sorted(unknown)}. "
                f"Valid columns: {WRITABLE_COLS}"
            )
        kw.setdefault("ts", time.time())
        kw.setdefault("degradation_level", 0)
        row = tuple(kw.get(col) for col in WRITABLE_COLS)

        with self._write_lock:
            self._pending.append(row)
            ready = len(self._pending) >= self._batch_size
        if ready:
            self.flush()

    def flush(self) -> None:
        with self._write_lock:
            if not self._pending:
                return
            batch, self._pending = self._pending, []
        self.conn.executemany(_INSERT_SQL, batch)
        self.conn.commit()

    # ---------------------------------------------------- keys table

    def upsert_keys(self, rows: Iterable[tuple]) -> None:
        """rows of (api_key_id, secret, tier, owner, created_ts)."""
        self.conn.executemany(
            "INSERT OR REPLACE INTO keys "
            "(api_key_id, secret, tier, owner, created_ts) VALUES (?,?,?,?,?)",
            list(rows),
        )
        self.conn.commit()

    def lookup_secret(self, secret: str):
        return self.conn.execute(
            "SELECT api_key_id, tier, owner FROM keys WHERE secret = ?", (secret,)
        ).fetchone()

    def all_keys(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM keys ORDER BY api_key_id").fetchall()

    def owners_df(self):
        """Ground truth, for SCORING ONLY. Never join this into a feature."""
        import pandas as pd

        return pd.read_sql_query("SELECT api_key_id, owner, tier FROM keys", self.conn)

    # ---------------------------------------------------- reading

    def read_df(self, run_id: str | None = None, decode: bool = True):
        """The only supported way to read the log.

        Returns a pandas DataFrame where:
            embedding      -> numpy array of 32 float32 (or None)
            model_probs    -> list of 10 floats (or None)
            returned_probs -> list of 10 floats (or None)

        Do not open the database yourself and parse the BLOB by hand.
        """
        import pandas as pd

        sql = "SELECT * FROM requests"
        params: list = []
        if run_id:
            sql += " WHERE (run_id = ? OR TRIM(run_id) = ?)"
            params.extend([run_id, run_id.strip()])
        sql += " ORDER BY request_id"

        df = pd.read_sql_query(sql, self.conn, params=params)
        if decode and len(df):
            # NULLs arrive from pandas as NaN (a float), not None, so test
            # the type rather than truthiness.
            df["embedding"] = df["embedding"].map(decode_embedding)
            df["model_probs"] = df["model_probs"].map(decode_probs)
            df["returned_probs"] = df["returned_probs"].map(decode_probs)
        return df

    def stats(self, run_id: str | None = None) -> dict:
        where, params = ("", [])
        if run_id:
            where, params = (" WHERE run_id = ?", [run_id])
        c = self.conn
        one = lambda sql: c.execute(sql, params).fetchone()["n"]  # noqa: E731
        return {
            "total_requests": one(f"SELECT COUNT(*) n FROM requests{where}"),
            "distinct_keys": one(
                f"SELECT COUNT(DISTINCT api_key_id) n FROM requests{where}"),
            "distinct_ips": one(
                f"SELECT COUNT(DISTINCT ip) n FROM requests{where}"),
            "error_responses": one(
                f"SELECT COUNT(*) n FROM requests{where}"
                + (" AND" if where else " WHERE") + " status_code >= 400"),
        }

    def read_timing_df(self, run_id: str | None = None, decode: bool = False):
        """Read requests log enriched with inter-query intervals (delta_t).

        Adds:
          delta_t: elapsed seconds since previous request by the SAME api_key_id.
                   (NaN for the key's first request)
          global_delta_t: elapsed seconds since previous request across the ENTIRE service.
                          (NaN for the first overall request)
        """
        import pandas as pd

        df = self.read_df(run_id=run_id, decode=decode)
        if df.empty:
            df["delta_t"] = pd.Series(dtype=float)
            df["global_delta_t"] = pd.Series(dtype=float)
            return df

        # Sort by ts and request_id to guarantee chronological order
        df = df.sort_values(["ts", "request_id"]).reset_index(drop=True)
        df["delta_t"] = df.groupby("api_key_id")["ts"].diff()
        df["global_delta_t"] = df["ts"].diff()
        return df

    def timing_stats(self, run_id: str | None = None) -> dict[str, Any]:
        """Summary time criteria for a run: duration, inter-arrival stats, and query rates."""
        tdf = self.read_timing_df(run_id=run_id, decode=False)
        if tdf.empty:
            return {
                "total_requests": 0,
                "duration_s": 0.0,
                "duration_min": 0.0,
                "mean_account_delta_t": 0.0,
                "median_account_delta_t": 0.0,
                "mean_global_delta_t": 0.0,
                "median_global_delta_t": 0.0,
                "effective_qps": 0.0,
            }

        start_ts = float(tdf["ts"].min())
        end_ts = float(tdf["ts"].max())
        duration_s = max(0.0, end_ts - start_ts)
        valid_deltas = tdf["delta_t"].dropna()
        valid_global = tdf["global_delta_t"].dropna()

        return {
            "total_requests": int(len(tdf)),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_s": duration_s,
            "duration_min": duration_s / 60.0,
            "mean_account_delta_t": float(valid_deltas.mean()) if len(valid_deltas) else 0.0,
            "median_account_delta_t": float(valid_deltas.median()) if len(valid_deltas) else 0.0,
            "mean_global_delta_t": float(valid_global.mean()) if len(valid_global) else 0.0,
            "median_global_delta_t": float(valid_global.median()) if len(valid_global) else 0.0,
            "effective_qps": float(len(tdf) / duration_s) if duration_s > 0 else float(len(tdf)),
        }

    def close(self) -> None:
        self.flush()
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None