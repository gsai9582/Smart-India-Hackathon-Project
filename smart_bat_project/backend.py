# backend.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import random
import asyncio

app = FastAPI(title="Smart Table Tennis Backend")

# Allow CORS so your frontend (likely served from file:// or localhost) can fetch
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class PlayerUpdate(BaseModel):
    score: Optional[int] = None
    strikes: Optional[int] = None
    misses: Optional[int] = None
    accuracy: Optional[int] = None

class MatchIn(BaseModel):
    match_id: str
    alex: int
    sarah: int
    winner: str

class MetaUpdate(BaseModel):
    matches_played: Optional[int] = None
    total_wins: Optional[int] = None

# ---------- In-memory state (mirrors your JS gameState) ----------
state = {
    "player1": {"id": 1, "name": "Alex Chen", "country": "China", "city": "Shanghai",
                "score": 18, "strikes": 35, "misses": 5, "accuracy": 88},
    "player2": {"id": 2, "name": "Sarah Kim", "country": "South Korea", "city": "Seoul",
                "score": 14, "strikes": 28, "misses": 9, "accuracy": 76},
    "scoreHistory": {"player1": [18], "player2": [14]},
    "accuracyHistory": {"player1": [88], "player2": [76]},
    "efficiencyHistory": {"player1": [88], "player2": [76]},
    "updateCount": 0,
    "matchesPlayed": 24,
    "totalWins": 18,
    "recentMatches": [
        {"match_id": "M101", "alex": 21, "sarah": 18, "winner": "Alex"},
        {"match_id": "M102", "alex": 19, "sarah": 21, "winner": "Sarah"},
        {"match_id": "M103", "alex": 22, "sarah": 20, "winner": "Alex"}
    ]
}

# Lock to protect concurrent updates (async)
state_lock = asyncio.Lock()

# ---------- Helper functions ----------
def compute_efficiency(strikes: int, misses: int) -> int:
    total = strikes + misses
    if total == 0:
        return 0
    return round((strikes / total) * 100)

async def push_history_limits(max_history: int = 10):
    # keep histories to max_history
    for k in ("player1", "player2"):
        if len(state["scoreHistory"][k]) > max_history:
            state["scoreHistory"][k] = state["scoreHistory"][k][-max_history:]
        if len(state["accuracyHistory"][k]) > max_history:
            state["accuracyHistory"][k] = state["accuracyHistory"][k][-max_history:]
        if len(state["efficiencyHistory"][k]) > max_history:
            state["efficiencyHistory"][k] = state["efficiencyHistory"][k][-max_history:]

# ---------- Endpoints ----------
@app.get("/api/state")
async def get_state():
    """Return the entire current state consumed by the frontend."""
    return state

@app.post("/api/player/{player_id}")
async def update_player(player_id: int, upd: PlayerUpdate):
    """Patch a player's stats. player_id = 1 (Alex) or 2 (Sarah)."""
    async with state_lock:
        key = "player1" if player_id == 1 else "player2" if player_id == 2 else None
        if key is None:
            raise HTTPException(status_code=404, detail="Player not found")
        p = state[key]
        if upd.score is not None:
            p["score"] = upd.score
        if upd.strikes is not None:
            p["strikes"] = upd.strikes
        if upd.misses is not None:
            p["misses"] = upd.misses
        if upd.accuracy is not None:
            p["accuracy"] = upd.accuracy
        # Recompute accuracy & efficiency if needed
        total = p["strikes"] + p["misses"]
        p["accuracy"] = round((p["strikes"] / total) * 100) if total > 0 else p["accuracy"]
        eff = compute_efficiency(p["strikes"], p["misses"])
        # append to histories
        state["scoreHistory"][key].append(p["score"])
        state["accuracyHistory"][key].append(p["accuracy"])
        state["efficiencyHistory"][key].append(eff)
        await push_history_limits()
    return {"status": "ok", "player": p}

@app.post("/api/simulate")
async def simulate_step():
    """Advance the match by one simulated step (same logic as your JS simulateLiveMatch)."""
    async with state_lock:
        # random chance updates (kept similar to frontend probabilities)
        if random.random() > 0.25:
            delta = random.randint(1, 3)
            state["player1"]["score"] += delta
            state["player1"]["strikes"] += random.randint(1, 4)
        if random.random() > 0.75:
            state["player1"]["misses"] += 1

        if random.random() > 0.35:
            state["player2"]["score"] += random.randint(1, 2)
            state["player2"]["strikes"] += random.randint(1, 3)
        if random.random() > 0.65:
            state["player2"]["misses"] += 1

        # recompute accuracy
        p1 = state["player1"]
        p2 = state["player2"]
        p1_total = p1["strikes"] + p1["misses"]
        p2_total = p2["strikes"] + p2["misses"]
        p1["accuracy"] = round((p1["strikes"] / p1_total) * 100) if p1_total > 0 else p1["accuracy"]
        p2["accuracy"] = round((p2["strikes"] / p2_total) * 100) if p2_total > 0 else p2["accuracy"]

        # push to histories
        state["scoreHistory"]["player1"].append(p1["score"])
        state["scoreHistory"]["player2"].append(p2["score"])
        state["accuracyHistory"]["player1"].append(p1["accuracy"])
        state["accuracyHistory"]["player2"].append(p2["accuracy"])
        state["efficiencyHistory"]["player1"].append(compute_efficiency(p1["strikes"], p1["misses"]))
        state["efficiencyHistory"]["player2"].append(compute_efficiency(p2["strikes"], p2["misses"]))

        # cap history length
        await push_history_limits(max_history=10)

        state["updateCount"] += 1

    return {"status": "ok", "state": state}

@app.get("/api/matches")
async def get_matches():
    """Return recent matches list."""
    return {"recentMatches": state["recentMatches"]}

@app.post("/api/matches")
async def add_match(m: MatchIn):
    """Add a recent match record."""
    async with state_lock:
        rec = {"match_id": m.match_id, "alex": m.alex, "sarah": m.sarah, "winner": m.winner}
        state["recentMatches"].insert(0, rec)
        # keep only last 50 matches
        if len(state["recentMatches"]) > 50:
            state["recentMatches"] = state["recentMatches"][:50]
        # increment counts (simple logic: if alex won, increment totalWins)
        state["matchesPlayed"] = state["matchesPlayed"] + 1 if state.get("matchesPlayed") is not None else 1
        if m.winner.lower() == "alex":
            state["totalWins"] = state.get("totalWins", 0) + 1
    return {"status": "ok", "recentMatches": state["recentMatches"]}

@app.post("/api/meta")
async def update_meta(meta: MetaUpdate):
    """Update matchesPlayed or totalWins manually."""
    async with state_lock:
        if meta.matches_played is not None:
            state["matchesPlayed"] = meta.matches_played
        if meta.total_wins is not None:
            state["totalWins"] = meta.total_wins
    return {"status": "ok", "meta": {"matchesPlayed": state["matchesPlayed"], "totalWins": state["totalWins"]}}
