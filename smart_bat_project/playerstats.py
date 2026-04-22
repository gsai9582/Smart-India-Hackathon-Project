# backend.py
import asyncio
import json
import sqlite3
import random
import time
from contextlib import closing
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional ML imports (install if you want ML)
try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import joblib
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "smart_bat.db"
MODEL_PATH = APP_DIR / "models"
MODEL_PATH.mkdir(exist_ok=True)

# --- Quick SQLite setup (simple tables for players, matches, settings, profile, reports) ---
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY,
            name TEXT,
            country TEXT,
            city TEXT,
            score INTEGER,
            strikes INTEGER,
            misses INTEGER,
            accuracy INTEGER,
            total_wins INTEGER DEFAULT 0
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT,
            p1_score INTEGER,
            p2_score INTEGER,
            winner TEXT,
            timestamp INTEGER
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            k TEXT PRIMARY KEY,
            v TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS profile (
            k TEXT PRIMARY KEY,
            v TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            created_at INTEGER
        )
        """)
        conn.commit()

init_db()

# ensure two example players exist (Alex, Sarah)
def ensure_default_players():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM players")
        if cur.fetchone()[0] == 0:
            cur.execute("""INSERT INTO players (name,country,city,score,strikes,misses,accuracy,total_wins)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        ("Alex Chen","China","Shanghai",18,35,5,88,18))
            cur.execute("""INSERT INTO players (name,country,city,score,strikes,misses,accuracy,total_wins)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        ("Sarah Kim","South Korea","Seoul",14,28,9,76,6))
            conn.commit()

ensure_default_players()

# --- App init ---
app = FastAPI(title="Smart Table Tennis Backend v1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Pydantic models for payloads ---
class PlayerUpdate(BaseModel):
    score: Optional[int] = None
    strikes: Optional[int] = None
    misses: Optional[int] = None
    accuracy: Optional[int] = None

class MatchIn(BaseModel):
    match_id: str
    p1_score: int
    p2_score: int
    winner: str

class SettingsItem(BaseModel):
    key: str
    value: str

class ReportIn(BaseModel):
    title: str
    content: str

# --- Simple WebSocket manager to broadcast live updates ---
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: Dict[str, Any]):
        dead = []
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for d in dead:
            self.disconnect(d)

manager = ConnectionManager()

# --- Utility DB helpers ---
def dict_from_row(cursor, row):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

def get_players() -> List[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM players")
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in rows]

def get_state_snapshot():
    players = get_players()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT match_id,p1_score,p2_score,winner,timestamp FROM matches ORDER BY id DESC LIMIT 10")
        recent = [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]
    return {"players": players, "recentMatches": recent, "timestamp": int(time.time())}

# --- REST endpoints ---

@app.get("/api/state")
async def api_state():
    """Return a snapshot used by frontend to populate dashboard"""
    return get_state_snapshot()

# Players CRUD
@app.get("/api/players")
async def api_players():
    return {"players": get_players()}

@app.get("/api/players/{player_id}")
async def api_get_player(player_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM players WHERE id = ?", (player_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Player not found")
        return dict(zip([d[0] for d in cur.description], row))

@app.post("/api/players/{player_id}")
async def api_update_player(player_id: int, upd: PlayerUpdate):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT strikes,misses FROM players WHERE id = ?", (player_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Player not found")
        strikes, misses = row
        # apply updates
        if upd.score is not None:
            cur.execute("UPDATE players SET score = ? WHERE id = ?", (upd.score, player_id))
        if upd.strikes is not None:
            cur.execute("UPDATE players SET strikes = ? WHERE id = ?", (upd.strikes, player_id))
            strikes = upd.strikes
        if upd.misses is not None:
            cur.execute("UPDATE players SET misses = ? WHERE id = ?", (upd.misses, player_id))
            misses = upd.misses
        # recompute accuracy
        if strikes + misses > 0:
            accuracy = round((strikes / (strikes + misses)) * 100)
            cur.execute("UPDATE players SET accuracy = ? WHERE id = ?", (accuracy, player_id))
        conn.commit()
    # broadcast updated state
    await manager.broadcast({"type": "player_update", "state": get_state_snapshot()})
    return {"status": "ok", "player": await api_get_player(player_id)}

# Matches endpoints
@app.get("/api/matches")
async def api_matches(limit: int = 20):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM matches ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return {"recentMatches": [dict(zip(cols, r)) for r in rows]}

@app.post("/api/matches")
async def api_add_match(m: MatchIn):
    ts = int(time.time())
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO matches (match_id,p1_score,p2_score,winner,timestamp) VALUES (?,?,?,?,?)",
                    (m.match_id, m.p1_score, m.p2_score, m.winner, ts))
        conn.commit()
        # update win count if winner is one of the players
        if m.winner.lower() in ("alex","alex chen"):
            cur.execute("UPDATE players SET total_wins = total_wins + 1 WHERE name LIKE 'Alex%'")
        elif m.winner.lower() in ("sarah","sarah kim"):
            cur.execute("UPDATE players SET total_wins = total_wins + 1 WHERE name LIKE 'Sarah%'")
        conn.commit()
    # notify clients
    await manager.broadcast({"type": "new_match", "match": m.dict(), "timestamp": ts, "state": get_state_snapshot()})
    return {"status": "ok"}

# Settings endpoints
@app.get("/api/settings")
async def api_get_settings():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT k,v FROM settings")
        return {"settings": {k:v for k,v in cur.fetchall()}}

@app.post("/api/settings")
async def api_set_setting(item: SettingsItem):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO settings (k,v) VALUES (?,?)", (item.key, item.value))
        conn.commit()
    return {"status": "ok"}

# Profile endpoints
@app.get("/api/profile")
async def api_get_profile():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT k,v FROM profile")
        return {"profile": {k:v for k,v in cur.fetchall()}}

@app.post("/api/profile")
async def api_set_profile(item: SettingsItem):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO profile (k,v) VALUES (?,?)", (item.key, item.value))
        conn.commit()
    return {"status": "ok"}

# Reports endpoints
@app.get("/api/reports")
async def api_get_reports(limit: int = 20):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id,title,content,created_at FROM reports ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return {"reports": [dict(zip(cols, r)) for r in rows]}

@app.post("/api/reports")
async def api_create_report(r: ReportIn):
    ts = int(time.time())
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO reports (title,content,created_at) VALUES (?,?,?)", (r.title, r.content, ts))
        conn.commit()
    return {"status": "ok"}

# Analytics endpoint (computes simple metrics from matches & players)
@app.get("/api/analytics")
async def api_analytics():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM matches")
        total_matches = cur.fetchone()[0]
        cur.execute("SELECT AVG(p1_score), AVG(p2_score) FROM matches")
        avg_scores = cur.fetchone()
        cur.execute("SELECT name,total_wins FROM players")
        players = [{"name": r[0], "wins": r[1]} for r in cur.fetchall()]

    return {
        "total_matches": total_matches,
        "average_scores": {"p1": avg_scores[0] or 0, "p2": avg_scores[1] or 0},
        "players": players
    }

# --- WebSocket endpoint for live updates ---
@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await manager.connect(ws)
    try:
        # send initial snapshot
        await ws.send_json({"type": "snapshot", "state": get_state_snapshot()})
        while True:
            # keep connection alive by awaiting messages (client can send pings/commands)
            msg = await ws.receive_text()
            # simple echo for now
            if msg == "ping":
                await ws.send_json({"type": "pong", "ts": int(time.time())})
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)

# --- Simulation background task that updates player stats periodically and broadcasts ---
_simulation_task: Optional[asyncio.Task] = None
SIM_INTERVAL = 2.0  # seconds

async def run_simulation():
    while True:
        # simple simulation: randomly update players
        await asyncio.sleep(SIM_INTERVAL)
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            # player 1 update
            if random.random() > 0.25:
                cur.execute("UPDATE players SET score = score + ? WHERE name LIKE 'Alex%'", (random.randint(1,3),))
                cur.execute("UPDATE players SET strikes = strikes + ? WHERE name LIKE 'Alex%'", (random.randint(1,4),))
            if random.random() > 0.75:
                cur.execute("UPDATE players SET misses = misses + 1 WHERE name LIKE 'Alex%'")
            # player 2 update
            if random.random() > 0.35:
                cur.execute("UPDATE players SET score = score + ? WHERE name LIKE 'Sarah%'", (random.randint(1,2),))
                cur.execute("UPDATE players SET strikes = strikes + ? WHERE name LIKE 'Sarah%'", (random.randint(1,3),))
            if random.random() > 0.65:
                cur.execute("UPDATE players SET misses = misses + 1 WHERE name LIKE 'Sarah%'")
            # recompute accuracies
            cur.execute("SELECT id,strikes,misses FROM players")
            for pid, strikes, misses in cur.fetchall():
                if strikes + misses > 0:
                    acc = round((strikes / (strikes + misses)) * 100)
                    cur.execute("UPDATE players SET accuracy = ? WHERE id = ?", (acc, pid))
            conn.commit()
        # broadcast new state
        await manager.broadcast({"type": "sim", "state": get_state_snapshot(), "ts": int(time.time())})

@app.on_event("startup")
async def startup_event():
    global _simulation_task
    if _simulation_task is None:
        _simulation_task = asyncio.create_task(run_simulation())

# --- ML train endpoint (optional) ---
@app.post("/api/train")
async def api_train(file: UploadFile = File(None)):
    """
    Train a simple model to predict match winner using uploaded CSV or DB matches.
    CSV expected columns: p1_score,p2_score,p1_strikes,p1_misses,p2_strikes,p2_misses,winner
    """
    if not ML_AVAILABLE:
        raise HTTPException(status_code=400, detail="ML libraries not installed. Install pandas scikit-learn joblib")

    # Load data: prefer uploaded CSV, else use DB matches (DB may lack strikes/misses -> we will synthesize)
    if file is not None:
        df = pd.read_csv(file.file)
    else:
        # try to build dataset from matches + players snapshots
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT p1_score,p2_score,winner FROM matches")
            rows = cur.fetchall()
        if not rows:
            # synthesize small dataset from current players
            with sqlite3.connect(DB_PATH) as conn:
                cur = conn.cursor()
                cur.execute("SELECT score,strikes,misses FROM players ORDER BY id ASC")
                pl = cur.fetchall()
            if len(pl) < 2:
                raise HTTPException(status_code=400, detail="Not enough data to train")
            # make synthetic rows
            rows = []
            for i in range(200):
                p1_score = max(0, pl[0][0] + random.randint(-5,5))
                p2_score = max(0, pl[1][0] + random.randint(-5,5))
                winner = "Alex" if p1_score >= p2_score else "Sarah"
                rows.append((p1_score,p2_score,winner))
            df = pd.DataFrame(rows, columns=["p1_score","p2_score","winner"])
    # Basic feature engineering
    if "winner" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain 'winner' column")
    # create simple features
    if "p1_strikes" not in df.columns:
        df["p1_strikes"] = df.get("p1_score", 0) * 1  # fallback heuristic
    if "p2_strikes" not in df.columns:
        df["p2_strikes"] = df.get("p2_score", 0) * 1
    X = df[["p1_score","p2_score","p1_strikes","p2_strikes"]].fillna(0)
    y = (df["winner"].str.lower() == "alex").astype(int)  # 1 = Alex wins, 0 = Sarah
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds, output_dict=True)
    # save model
    model_file = MODEL_PATH / f"rf_model_{int(time.time())}.joblib"
    joblib.dump(clf, model_file)
    return {"status": "ok", "accuracy": acc, "report": report, "model_file": str(model_file)}

# Endpoint to get path to latest model
@app.get("/api/models/latest")
async def api_models_latest():
    files = sorted(MODEL_PATH.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return {"latest": None}
    return {"latest": str(files[0])}

# health check
@app.get("/api/health")
async def api_health():
    return {"status": "ok", "ml_available": ML_AVAILABLE, "db": str(DB_PATH)}

# --- end of file ---
