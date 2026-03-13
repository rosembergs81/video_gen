"""
modules/database.py
────────────────────
Backend SQLite para persistencia de:
  • Historial de generaciones (parámetros + seed + output path)
  • Presets de prompts favoritos
  • Comparación de variaciones side-by-side
"""

from __future__ import annotations
import sqlite3
import json
import time
import queue
import contextlib
import threading
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Any, Optional


DB_PATH = Path("videogen.db")


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    REAL NOT NULL,
    model         TEXT NOT NULL,
    prompt        TEXT NOT NULL,
    negative      TEXT,
    params        TEXT,         -- JSON blob (frames, fps, steps, guidance, seed…)
    loras         TEXT,         -- JSON list of active LoRA display names
    motion_tags   TEXT,         -- JSON (camera, subject, speed, style)
    output_path   TEXT,
    duration_s    REAL,
    frame_count   INTEGER,
    tags          TEXT          -- user-assigned tags JSON list
);

CREATE TABLE IF NOT EXISTS presets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,
    category    TEXT DEFAULT 'general',
    created_at  REAL NOT NULL,
    model       TEXT,
    prompt      TEXT,
    negative    TEXT,
    params      TEXT,   -- JSON
    motion_tags TEXT,   -- JSON
    loras       TEXT,   -- JSON
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS comparisons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL NOT NULL,
    title       TEXT,
    gen_ids     TEXT   -- JSON list of generation IDs
);

CREATE INDEX IF NOT EXISTS idx_gen_created  ON generations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gen_model    ON generations(model);
CREATE INDEX IF NOT EXISTS idx_preset_name  ON presets(name);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationRecord:
    model:        str
    prompt:       str
    negative:     str       = ""
    params:       dict      = field(default_factory=dict)
    loras:        list      = field(default_factory=list)
    motion_tags:  dict      = field(default_factory=dict)
    output_path:  str       = ""
    duration_s:   float     = 0.0
    frame_count:  int       = 0
    tags:         list[str] = field(default_factory=list)
    id:           int       = 0
    created_at:   float     = 0.0


@dataclass
class Preset:
    name:       str
    category:   str  = "general"
    model:      str  = ""
    prompt:     str  = ""
    negative:   str  = ""
    params:     dict = field(default_factory=dict)
    motion_tags:dict = field(default_factory=dict)
    loras:      list = field(default_factory=list)
    notes:      str  = ""
    id:         int  = 0
    created_at: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────────────────

class VideoGenDB:
    """
    Thread-safe SQLite backend.

    Usage:
        db = VideoGenDB()
        gen_id = db.save_generation(GenerationRecord(model="...", prompt="..."))
        history = db.get_history(limit=20)
        db.save_preset(Preset(name="My preset", prompt="..."))
    """

    def __init__(self, db_path: Path = DB_PATH, pool_size: int = 5):
        self.db_path = db_path
        self._pool = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")   # better concurrency
            self._pool.put(conn)
        self._init_db()

    @contextlib.contextmanager
    def get_connection(self):
        """Return a pooled connection."""
        conn = self._pool.get()
        try:
            with conn: # implicit transaction
                yield conn
        finally:
            self._pool.put(conn)

    def _init_db(self):
        with self.get_connection() as conn:
            conn.executescript(_SCHEMA)

    def close(self):
        """Close all connections in the pool."""
        while not self._pool.empty():
            conn = self._pool.get()
            conn.close()

    # ── Generations ──────────────────────────────────────────────────────────

    def save_generation(self, rec: GenerationRecord) -> int:
        rec.created_at = time.time()
        with self.get_connection() as conn:
            cur = conn.execute("""
                INSERT INTO generations
                    (created_at, model, prompt, negative, params, loras,
                     motion_tags, output_path, duration_s, frame_count, tags)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                rec.created_at, rec.model, rec.prompt, rec.negative,
                json.dumps(rec.params),
                json.dumps(rec.loras),
                json.dumps(rec.motion_tags),
                rec.output_path,
                rec.duration_s,
                rec.frame_count,
                json.dumps(rec.tags),
            ))
            return cur.lastrowid

    def get_history(
        self,
        limit: int = 50,
        model: Optional[str] = None,
        search: Optional[str] = None,
    ) -> list[GenerationRecord]:
        conditions, args = [], []
        if model:
            conditions.append("model = ?"); args.append(model)
        if search:
            conditions.append("prompt LIKE ?"); args.append(f"%{search}%")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        args.append(limit)

        with self.get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM generations {where} ORDER BY created_at DESC LIMIT ?",
                args,
            ).fetchall()

        return [self._row_to_gen(r) for r in rows]

    def get_generation(self, gen_id: int) -> Optional[GenerationRecord]:
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM generations WHERE id = ?", (gen_id,)
            ).fetchone()
        return self._row_to_gen(row) if row else None

    def delete_generation(self, gen_id: int):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM generations WHERE id = ?", (gen_id,))

    def add_tag(self, gen_id: int, tag: str):
        rec = self.get_generation(gen_id)
        if rec:
            if tag not in rec.tags:
                rec.tags.append(tag)
            with self.get_connection() as conn:
                conn.execute(
                    "UPDATE generations SET tags = ? WHERE id = ?",
                    (json.dumps(rec.tags), gen_id),
                )

    def _row_to_gen(self, row) -> GenerationRecord:
        return GenerationRecord(
            id=row["id"],
            created_at=row["created_at"],
            model=row["model"],
            prompt=row["prompt"],
            negative=row["negative"] or "",
            params=json.loads(row["params"] or "{}"),
            loras=json.loads(row["loras"] or "[]"),
            motion_tags=json.loads(row["motion_tags"] or "{}"),
            output_path=row["output_path"] or "",
            duration_s=row["duration_s"] or 0.0,
            frame_count=row["frame_count"] or 0,
            tags=json.loads(row["tags"] or "[]"),
        )

    # ── Presets ──────────────────────────────────────────────────────────────

    def save_preset(self, preset: Preset) -> int:
        preset.created_at = time.time()
        with self.get_connection() as conn:
            try:
                cur = conn.execute("""
                    INSERT INTO presets
                        (name, category, created_at, model, prompt, negative,
                         params, motion_tags, loras, notes)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (
                    preset.name, preset.category, preset.created_at,
                    preset.model, preset.prompt, preset.negative,
                    json.dumps(preset.params),
                    json.dumps(preset.motion_tags),
                    json.dumps(preset.loras),
                    preset.notes,
                ))
                return cur.lastrowid
            except sqlite3.IntegrityError:
                # Update existing
                conn.execute("""
                    UPDATE presets SET
                        category=?, model=?, prompt=?, negative=?,
                        params=?, motion_tags=?, loras=?, notes=?
                    WHERE name=?
                """, (
                    preset.category, preset.model, preset.prompt, preset.negative,
                    json.dumps(preset.params),
                    json.dumps(preset.motion_tags),
                    json.dumps(preset.loras),
                    preset.notes,
                    preset.name,
                ))
                return self.get_preset(preset.name).id

    def get_preset(self, name: str) -> Optional[Preset]:
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM presets WHERE name = ?", (name,)
            ).fetchone()
        return self._row_to_preset(row) if row else None

    def list_presets(self, category: Optional[str] = None) -> list[Preset]:
        if category:
            with self.get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM presets WHERE category=? ORDER BY name",
                    (category,)
                ).fetchall()
        else:
            with self.get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM presets ORDER BY category, name"
                ).fetchall()
        return [self._row_to_preset(r) for r in rows]

    def delete_preset(self, name: str):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM presets WHERE name = ?", (name,))

    def _row_to_preset(self, row) -> Preset:
        return Preset(
            id=row["id"],
            created_at=row["created_at"],
            name=row["name"],
            category=row["category"] or "general",
            model=row["model"] or "",
            prompt=row["prompt"] or "",
            negative=row["negative"] or "",
            params=json.loads(row["params"] or "{}"),
            motion_tags=json.loads(row["motion_tags"] or "{}"),
            loras=json.loads(row["loras"] or "[]"),
            notes=row["notes"] or "",
        )

    # ── Comparisons ──────────────────────────────────────────────────────────

    def save_comparison(self, title: str, gen_ids: list[int]) -> int:
        with self.get_connection() as conn:
            cur = conn.execute(
                "INSERT INTO comparisons (created_at, title, gen_ids) VALUES (?,?,?)",
                (time.time(), title, json.dumps(gen_ids)),
            )
            return cur.lastrowid

    def get_comparison(self, cmp_id: int) -> Optional[dict]:
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM comparisons WHERE id = ?", (cmp_id,)
            ).fetchone()
        if not row:
            return None
        gen_ids = json.loads(row["gen_ids"])
        generations = [self.get_generation(gid) for gid in gen_ids]
        return {
            "id":          row["id"],
            "title":       row["title"],
            "created_at":  row["created_at"],
            "generations": [g for g in generations if g],
        }

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self.get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
            by_model = conn.execute(
                "SELECT model, COUNT(*) as n FROM generations GROUP BY model"
            ).fetchall()
            avg_dur = conn.execute(
                "SELECT AVG(duration_s) FROM generations WHERE duration_s > 0"
            ).fetchone()[0]
            preset_count = conn.execute("SELECT COUNT(*) FROM presets").fetchone()[0]
        return {
            "total_generations": total,
            "by_model":          {r["model"]: r["n"] for r in by_model},
            "avg_duration_s":    round(avg_dur or 0, 1),
            "total_presets":     preset_count,
        }

    # ── UI helpers ────────────────────────────────────────────────────────────

    def history_as_table(self, limit: int = 50) -> list[list]:
        """For Gradio gr.DataFrame."""
        import datetime
        records = self.get_history(limit=limit)
        rows = []
        for r in records:
            dt = datetime.datetime.fromtimestamp(r.created_at).strftime("%Y-%m-%d %H:%M")
            rows.append([
                r.id,
                dt,
                r.model.split("(")[0].strip(),
                r.prompt[:60] + "…" if len(r.prompt) > 60 else r.prompt,
                r.params.get("seed", "?"),
                f"{r.duration_s:.1f}s",
                r.frame_count,
                ", ".join(r.tags) if r.tags else "",
                r.output_path,
            ])
        return rows

    HISTORY_COLUMNS = [
        "ID", "Fecha", "Modelo", "Prompt", "Seed",
        "Duración", "Frames", "Tags", "Archivo",
    ]
