from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy data for illustration
players_data = [
    {"name": "Alex Chen", "score": 18, "strikes": 35, "misses": 5, "accuracy": 88},
    {"name": "Sarah Kim", "score": 14, "strikes": 28, "misses": 9, "accuracy": 76}
]

settings_data = {
    "theme": "light",
    "notifications": True
}

reports_data = [
    {"id": 1, "title": "Monthly Performance", "content": "Report content here..."}
]

analytics_data = {
    "totalMatches": 120,
    "averageScore": 16.5
}

# Player Stats API
@app.get("/api/players")
async def get_players():
    return {"players": players_data}

# Settings API
@app.get("/api/settings")
async def get_settings():
    return settings_data

# Reports API
@app.get("/api/reports")
async def get_reports():
    return {"reports": reports_data}

# Analytics API
@app.get("/api/analytics")
async def get_analytics():
    return analytics_data

# If you have any other endpoints or logic, add them similarly here.
