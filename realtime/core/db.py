"""Lưu phiên chạy và sự kiện vào SQLite (CLAUDE.md mục 5.5)."""
import json
import re
import sqlite3
import threading
import unicodedata
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL, ended_at TEXT, source TEXT, config_json TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    ts TEXT NOT NULL, type TEXT NOT NULL, severity INTEGER,
    track_ids TEXT,              -- JSON, vd "[3, 7]"
    confidence REAL, snapshot_path TEXT, clip_path TEXT,
    reviewed INTEGER DEFAULT 0,
    is_true INTEGER,             -- 1 đúng, 0 báo nhầm, NULL chưa duyệt
    note TEXT);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
"""


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def slug(s):
    """'TÉ NGÃ RỒI NẰM BẤT ĐỘNG' -> 'te_nga_roi_nam_bat_dong' (dùng cho tên file)."""
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


class EventDB:
    def __init__(self, path="realtime/events/events.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
            cols = {r[1] for r in self.conn.execute("PRAGMA table_info(events)")}
            for c, typ in (("scene_score", "REAL"), ("scene_group", "TEXT"), ("fight_score", "REAL")):
                if c not in cols:                           # DB cu: them cot luong 2 / luong 1
                    self.conn.execute(f"ALTER TABLE events ADD COLUMN {c} {typ}")
            self.conn.commit()

    def _exec(self, sql, args=()):
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur

    def _query(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def start_session(self, source, config=None):
        cfg = json.dumps(config, ensure_ascii=False) if config is not None else None
        return self._exec("INSERT INTO sessions(started_at, source, config_json) VALUES (?, ?, ?)",
                          (now_str(), str(source), cfg)).lastrowid

    def end_session(self, session_id):
        self._exec("UPDATE sessions SET ended_at = ? WHERE id = ?", (now_str(), session_id))

    def add_event(self, session_id, type, severity, ids, confidence=None, snapshot=None, clip=None,
                  scene_score=None, scene_group=None, fight_score=None):
        return self._exec(
            "INSERT INTO events(session_id, ts, type, severity, track_ids, confidence, snapshot_path, clip_path,"
            " scene_score, scene_group, fight_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, now_str(), type, severity, json.dumps([int(i) for i in ids]),
             None if confidence is None else float(confidence),
             None if snapshot is None else str(snapshot), None if clip is None else str(clip),
             None if scene_score is None else float(scene_score), scene_group,
             None if fight_score is None else float(fight_score))).lastrowid

    def set_clip(self, event_id, path):
        self._exec("UPDATE events SET clip_path = ? WHERE id = ?", (str(path), event_id))

    def review(self, event_id, is_true, note=None):
        self._exec("UPDATE events SET reviewed = 1, is_true = ?, note = ? WHERE id = ?",
                   (int(bool(is_true)), note, event_id))

    def unreviewed(self):
        return self._query("SELECT * FROM events WHERE reviewed = 0 ORDER BY id")

    def stats(self):
        """Theo từng loại: total, reviewed, correct, precision (= correct / reviewed, None nếu chưa duyệt)."""
        rows = self._query(
            "SELECT type, COUNT(*) AS total, SUM(reviewed) AS reviewed,"
            " SUM(CASE WHEN is_true = 1 THEN 1 ELSE 0 END) AS correct"
            " FROM events GROUP BY type ORDER BY total DESC")
        for r in rows:
            r["precision"] = r["correct"] / r["reviewed"] if r["reviewed"] else None
        return rows

    def close(self):
        with self.lock:
            self.conn.close()
