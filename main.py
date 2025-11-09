# backend/main.py
import os
import uuid
import datetime
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from formation_service import process_logo

# Initialize FastAPI
app = FastAPI(title="Drone Formation Orchestrator")

# ✅ Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories and DB setup
BACKEND_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BACKEND_DIR, "data")
DB_PATH = os.path.join(BACKEND_DIR, "jobs_db.json")
os.makedirs(DATA_DIR, exist_ok=True)

# Ensure DB exists
if not os.path.exists(DB_PATH):
    with open(DB_PATH, "w") as f:
        json.dump([], f)

# Utility functions
def load_jobs():
    with open(DB_PATH, "r") as f:
        return json.load(f)

def save_job(job):
    jobs = load_jobs()
    jobs.append(job)
    with open(DB_PATH, "w") as f:
        json.dump(jobs, f, indent=2)

# ===============================
# 📤 API Endpoints
# ===============================

@app.post("/upload-logo")
async def upload_logo(file: UploadFile = File(...), num_points: int = 5):
    """
    Upload a logo image.
    Extracts formation coordinates and stores them.
    Returns job record with coords and saved target file path.
    """
    job_id = str(uuid.uuid4())
    saved_image = os.path.join(DATA_DIR, f"{job_id}_{file.filename}")

    # Save uploaded image
    with open(saved_image, "wb") as f:
        f.write(await file.read())

    try:
        targets_file, coords = process_logo(saved_image, num_points=num_points)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    job = {
        "job_id": job_id,
        "filename": file.filename,
        "image_saved": os.path.relpath(saved_image, BACKEND_DIR),
        "targets_file": os.path.relpath(targets_file, BACKEND_DIR),
        "coords": coords.tolist(),
        "status": "completed",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }

    save_job(job)
    return {"job_id": job_id, "coords": coords.tolist(), "targets_file": job["targets_file"]}

@app.get("/status/{job_id}")
def status(job_id: str):
    jobs = load_jobs()
    for j in jobs:
        if j["job_id"] == job_id:
            return j
    return JSONResponse(status_code=404, content={"error": "job not found"})

@app.get("/history")
def history():
    return load_jobs()

@app.post("/generate-show/{job_id}")
def generate_show(job_id: str):
    """
    Placeholder: would run orchestration / RL inference to produce timed trajectories.
    Right now returns the targets for the given job_id.
    """
    jobs = load_jobs()
    for j in jobs:
        if j["job_id"] == job_id:
            return {
                "message": "generate_show (simulated)",
                "targets_file": j["targets_file"],
                "coords": j["coords"]
            }
    return JSONResponse(status_code=404, content={"error": "job not found"})
