from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List
import uvicorn
import logging
import cv2
import time
import os
import signal
import subprocess
import sys
from core.streaming import FrameBridge
from typing import Dict, Optional
import platform

# Process management for recognition engine
engine_processes: Dict[int, subprocess.Popen] = {}
from api import schemas
from api.dependencies import get_db

# Import routers
from api.routes import students, attendance


app = FastAPI(
    title="EMSVIA Attendance API",
    description="Backend API for the Industrial-Grade Face Recognition Attendance System",
    version="1.0.0"
)

@app.on_event("shutdown")
def _shutdown_cleanup():
    """
    Ensure camera devices are released when the API stops.

    The recognition engines are launched as separate OS processes, so if the API
    exits without stopping them, they will keep running and hold /dev/video* open.
    """
    global engine_processes
    for cam_id, process in list(engine_processes.items()):
        try:
            # Kill the entire process group (we start engines with setsid)
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass
        finally:
            engine_processes.pop(cam_id, None)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to Streamlit's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to EMSVIA Attendance API",
        "docs": "/docs",
        "status": "online"
    }

@app.get("/api/stats", response_model=schemas.SystemStats)
def get_system_stats(db: Session = Depends(get_db)):
    """Retrieve high-level system statistics for the dashboard."""
    from database.models import Student, AttendanceRecord, UnknownFace
    from datetime import datetime, time
    from api.dependencies import get_db_manager
    from models.gpu_manager import GPUModelManager
    
    db_manager = get_db_manager()
    gpu_mgr = GPUModelManager()
    
    # Calculate today's start
    today_start = datetime.combine(datetime.now().date(), time.min)
    
    total_students = db.query(Student).count()
    attendance_today = db.query(AttendanceRecord).filter(AttendanceRecord.timestamp >= today_start).count()
    unknown_pending = db.query(UnknownFace).filter(UnknownFace.reviewed == False).count()
    
    return {
        "total_students": total_students,
        "attendance_today": attendance_today,
        "unknown_faces_pending": unknown_pending,
        "gpu_active": gpu_mgr.is_gpu_ready()
    }


@app.get("/api/unknown/pending", response_model=List[schemas.UnknownFaceResponse])
def get_pending_unknowns(db: Session = Depends(get_db)):
    """Retrieve unreviewed unknown faces."""
    from database.models import UnknownFace
    return db.query(UnknownFace).filter(UnknownFace.reviewed == False).order_by(UnknownFace.timestamp.desc()).limit(10).all()

@app.patch("/api/unknown/{face_id}/review", response_model=bool)
def review_unknown_face(face_id: int, db: Session = Depends(get_db)):
    """Mark an unknown face as reviewed."""
    from database.models import UnknownFace
    face = db.query(UnknownFace).filter(UnknownFace.id == face_id).first()
    if not face:
        raise HTTPException(status_code=404, detail="Face not found")
    face.reviewed = True
    db.commit()
    return True

@app.get("/api/cameras/stream")
async def video_stream(cam_id: int = 0):
    """MJPEG Video Stream with specified camera index."""
    bridge = FrameBridge(f"emsvia_cam_{cam_id}")
    
    def gen_frames(cid):
        # Try Bridge first (AI-Processed feed)
        last_meta_check = 0
        last_raw_open_attempt = 0.0
        raw_open_backoff_s = 5.0
        
        while True:
            # Check if engine is alive every 1 second
            meta = bridge.get_meta()
            engine_running = meta and (time.time() - meta.get('timestamp', 0) < 3.0)
            
            if engine_running:
                frame_bytes, _ = bridge.get()
                if frame_bytes:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(0.03) # ~30 FPS
                    continue
            
            # Fallback to Raw Feed
            # Avoid OpenCV spam when the device node doesn't exist (common on Linux).
            if platform.system().lower() == "linux":
                if not os.path.exists(f"/dev/video{cid}"):
                    time.sleep(1.0)
                    continue

            # Backoff raw open attempts so we don't spam stderr.
            now = time.time()
            if now - last_raw_open_attempt < raw_open_backoff_s:
                time.sleep(0.5)
                continue
            last_raw_open_attempt = now

            cap = cv2.VideoCapture(cid)
            if not cap.isOpened():
                time.sleep(1.0)
                continue
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            while True:
                # Check if engine started while we were streaming raw
                if time.time() - last_meta_check > 1.0:
                    meta = bridge.get_meta()
                    if meta and (time.time() - meta.get('timestamp', 0) < 3.0):
                        break # Switch back to bridge
                    last_meta_check = time.time()
                
                success, frame = cap.read()
                if not success: break
                
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.04)
            
            cap.release()
            
    return StreamingResponse(gen_frames(cam_id), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/api/engine/status")
async def engine_status():
    """Check the status of all running recognition engines."""
    results = {}
    from core.streaming import FrameBridge
    
    # Collect status from all up to 10 cameras by checking bridges directly
    for cam_id in range(10):
        bridge = FrameBridge(f"emsvia_cam_{cam_id}")
        meta = bridge.get_meta()
        
        if meta and (time.time() - meta.get('timestamp', 0) < 3.0):
            results[cam_id] = {
                "status": "running",
                "fps": meta.get('fps', 0),
                "gpu_summary": meta.get('gpu', 'N/A'),
                "camera_id": meta.get('camera_id', cam_id),
                "role": meta.get('role', 'entry'),
                "active_tracks": meta.get("active_tracks", 0)
            }
        elif cam_id in engine_processes and engine_processes[cam_id].poll() is None:
            results[cam_id] = {"status": "starting"}
            
    # Clean up dead processes from our tracking list
    for cam_id, process in list(engine_processes.items()):
        if process.poll() is not None:
            del engine_processes[cam_id]
            
    return {"engines": results}            
    return {"engines": results}

@app.post("/api/engine/start")
async def start_engine(cam_id: int = 0, role: str = "entry"):
    """Launch the recognition engine process on a specific camera."""
    try:
        # Check if already running for this camera
        if cam_id in engine_processes and engine_processes[cam_id].poll() is None:
            return {"status": "already_running", "camera_id": cam_id}
            
        # Ensure log directory exists
        from config.settings import LOGS_DIR
        startup_log = LOGS_DIR / f"engine_cam_{cam_id}.log"
        log_file = open(startup_log, "w")
        
        # Ensure cleanup of old shared files for this cam
        bridge_name = f"emsvia_cam_{cam_id}"
        FrameBridge(bridge_name).clear()
        
        # Pass environment variables explicitly to ensure CUDA/LD_LIBRARY_PATH are preserved
        env = os.environ.copy()
        
        cmd = [sys.executable, "main_gpu.py", "--headless", "--camera", str(cam_id), "--role", role, "--bridge", bridge_name]
        
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            cwd=os.getcwd(),
            env=env,
            preexec_fn=os.setsid
        )
        
        engine_processes[cam_id] = process
        logger.info(f"Recognition engine started for camera {cam_id} (Role: {role})")
        return {"status": "started", "camera_id": cam_id, "role": role}
    except Exception as e:
        logger.error(f"Failed to start engine for camera {cam_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/engine/stop")
async def stop_engine(cam_id: int = 0):
    """Stop the recognition engine for a specific camera."""
    if cam_id in engine_processes:
        process = engine_processes[cam_id]
        try:
            # Kill the entire process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception:
            process.terminate()
        del engine_processes[cam_id]
        
    # Fallback: kill any zombie/orphaned scripts for this camera
    try:
        subprocess.run(["pkill", "-f", f"main_gpu.py.*--camera {cam_id}"], check=False)
    except Exception:
        pass
        
    # Give it a moment to stop then clear bridge
    time.sleep(0.5)
    FrameBridge(f"emsvia_cam_{cam_id}").clear()
    
    return {"status": "stopped", "camera_id": cam_id}


@app.get("/api/cameras/detect")
def detect_cameras():
    """Scan and return working camera indices."""
    working = []
    # Scan up to 10 indices
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    working.append({
                        "index": i,
                        "name": f"Camera {i}"
                    })
                cap.release()
        except:
            continue
    return working

@app.post("/api/embeddings/rebuild")
async def rebuild_embeddings():
    """Trigger the embedding generation script."""
    import subprocess
    import sys
    try:
        # Run scripts/generate_embeddings.py as a separate process to avoid blocking
        # and to ensure a clean model context
        process = subprocess.Popen(
            [sys.executable, "scripts/generate_embeddings.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )
        return {"status": "processing", "pid": process.pid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def get_health():
    """Retrieve detailed hardware health stats."""
    from models.gpu_manager import GPUModelManager
    gpu_mgr = GPUModelManager()
    
    return {
        "gpu": gpu_mgr.monitor_memory(),
        "system": {
            "status": "online",
            "version": "2.1.0-gpu",
            "engine": "Async-TensorFlow-XLA"
        }
    }

@app.get("/api/system/logs")
def get_system_logs():
    """Read the last 50 lines of the system logs."""
    from config.settings import LOG_FILE
    import os
    if not os.path.exists(LOG_FILE):
        return {"logs": ["Log file not found."]}
    
    try:
        with open(LOG_FILE, "r") as f:
            # Efficiently read last 50 lines
            lines = f.readlines()
            return {"logs": [line.strip() for line in lines[-50:]]}
    except Exception as e:
        return {"logs": [f"Error reading logs: {str(e)}"]}

@app.get("/api/students/presence")
def get_students_presence(db: Session = Depends(get_db)):
    """Analyze today's attendance to determine student presence status."""
    from database.models import Student, AttendanceRecord
    from datetime import datetime, time as dt_time, timedelta
    
    # 1. Setup time range for today
    today_start = datetime.combine(datetime.now().date(), dt_time.min)
    today_end = datetime.combine(datetime.now().date(), dt_time.max)
    
    # 2. Get all students
    students = db.query(Student).filter(Student.is_active == True).all()
    
    # 3. Required stay (1.5 hours)
    REQ_MINUTES = 90
    
    results = []
    for student in students:
        # Get today's records for this student
        records = db.query(AttendanceRecord).filter(
            AttendanceRecord.student_id == student.id,
            AttendanceRecord.timestamp >= today_start,
            AttendanceRecord.timestamp <= today_end
        ).order_by(AttendanceRecord.timestamp.asc()).all()
        
        status = "absent"
        entry_time = None
        exit_time = None
        duration_mins = 0
        
        if records:
            # Find first entry
            entries = [r for r in records if r.entry_type == "entry"]
            if entries:
                entry_time = entries[0].timestamp
                status = "in_school"
                
                # Find last exit (must be after first entry)
                exits = [r for r in records if r.entry_type == "exit" and r.timestamp > entry_time]
                if exits:
                    exit_time = exits[-1].timestamp
                    duration = exit_time - entry_time
                    duration_mins = int(duration.total_seconds() / 60)
                    
                    if duration_mins >= REQ_MINUTES:
                        status = "present"
                    else:
                        status = "under_time"
        
        results.append({
            "id": student.student_id,
            "name": f"{student.first_name} {student.last_name}",
            "status": status,
            "entry_time": entry_time.strftime("%H:%M:%S") if entry_time else "N/A",
            "exit_time": exit_time.strftime("%H:%M:%S") if exit_time else "N/A",
            "duration": f"{duration_mins} mins" if duration_mins > 0 else "N/A"
        })
        
    return results

@app.delete("/api/unknown/{face_id}", status_code=204)
def delete_unknown_face(face_id: int, db: Session = Depends(get_db)):
    """Delete an unknown face record."""
    from database.models import UnknownFace
    face = db.query(UnknownFace).filter(UnknownFace.id == face_id).first()
    if not face:
        raise HTTPException(status_code=404, detail="Face not found")
    db.delete(face)
    db.commit()
    return None






# Include routers
app.include_router(students.router, prefix="/api/students", tags=["Students"])
app.include_router(attendance.router, prefix="/api/attendance", tags=["Attendance"])


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
