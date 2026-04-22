import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta, time

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from config.settings import DATABASE_URL, UNKNOWN_FACES_DIR, LOG_FILE
from database.crud import AttendanceDB
from models.gpu_manager import GPUModelManager

# Initialize components
db_manager = AttendanceDB(DATABASE_URL)
gpu_mgr = GPUModelManager()

# --- Helper Functions ---

def load_data(session):
    """Load core data for the dashboard."""
    students = db_manager.get_all_students(session)
    history = db_manager.get_attendance_history(session)
    unknowns = db_manager.get_unreviewed_faces(session)
    return students, history, unknowns

def render_overview(session):
    st.header("📈 System Overview")
    
    students, history, unknowns = load_data(session)
    
    # Hero Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Enrolled", len(students))
    
    # Today's attendance
    today_start = datetime.combine(datetime.now().date(), time.min)
    today_count = sum(1 for r in history if r.timestamp >= today_start and r.status == "present")
    col2.metric("Present Today", today_count)
    
    col3.metric("Pending Reviews", len(unknowns), delta=len(unknowns), delta_color="inverse")
    
    gpu_status = "Active" if gpu_mgr.is_gpu_ready() else "CPU Mode"
    col4.metric("GPU Engine", gpu_status)

    # Charts Section
    st.divider()
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Attendance Trends (Last 7 Days)")
        if history:
            df = pd.DataFrame([{"date": r.timestamp.date(), "id": r.id} for r in history])
            df_counts = df.groupby("date").count().reset_index()
            fig = px.area(df_counts, x="date", y="id", labels={"id": "Logs", "date": "Date"},
                         color_discrete_sequence=["#00CC96"])
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for trends.")

    with c2:
        st.subheader("Status Distribution")
        if history:
            status_counts = pd.DataFrame([{"status": r.status} for r in history]).status.value_counts()
            fig = px.pie(values=status_counts.values, names=status_counts.index, 
                         hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No logs yet.")

def render_students(session):
    st.header("👥 Student Management")
    
    # Add Student Form
    with st.expander("➕ Register New Student"):
        with st.form("add_student"):
            c1, c2 = st.columns(2)
            s_id = c1.text_input("Student ID (Unique)", placeholder="STU001")
            first = c1.text_input("First Name")
            last = c2.text_input("Last Name")
            email = c2.text_input("Email")
            
            submitted = st.form_submit_button("Register Student")
            if submitted:
                if s_id and first and last:
                    new_s = db_manager.add_student(session, s_id, first, last, email)
                    if new_s:
                        st.success(f"Student {first} {last} registered!")
                        st.rerun()
                else:
                    st.error("Please fill required fields (ID, First, Last)")

    # Student List
    students = db_manager.get_all_students(session)
    if students:
        df = pd.DataFrame([{
            "Student ID": s.student_id,
            "Name": f"{s.first_name} {s.last_name}",
            "Email": s.email,
            "Enrolled": s.enrollment_date.strftime("%Y-%m-%d"),
            "Status": "Active" if s.is_active else "Inactive"
        } for s in students])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No students enrolled.")

def render_attendance(session):
    st.header("📅 Attendance Logs & Corrections")
    
    # Filter
    students = db_manager.get_all_students(session)
    s_names = {s.id: f"{s.first_name} {s.last_name} ({s.student_id})" for s in students}
    
    selected_s = st.selectbox("Filter by Student", ["All"] + list(s_names.values()))
    
    history = db_manager.get_attendance_history(session)
    if history:
        # Filter logic
        filtered = history
        if selected_s != "All":
            s_id_internal = [k for k, v in s_names.items() if v == selected_s][0]
            filtered = [r for r in history if r.student_id == s_id_internal]
            
        df = pd.DataFrame([{
            "ID": h.id,
            "Time": h.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Student": s_names.get(h.student_id, "Unknown"),
            "Status": h.status,
            "Conf": f"{h.confidence_score:.2f}",
            "Camera": h.camera_id
        } for h in filtered])
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Correction Interface
        st.divider()
        st.subheader("Manual Correction")
        col1, col2, col3 = st.columns([1, 1, 1])
        record_id = col1.number_input("Record ID to Update", min_value=1, step=1)
        new_status = col2.selectbox("New Status", ["present", "late", "excused", "absent", "departed"])
        
        if col3.button("Update Record"):
            success = db_manager.update_attendance_record(session, int(record_id), {"status": new_status})
            if success:
                st.success(f"Record {record_id} updated to {new_status}")
                st.rerun()
            else:
                st.error("Record not found")
    else:
        st.info("No records found.")

def render_unknowns(session):
    st.header("🔍 Unknown Face Review Queue")
    unknowns = db_manager.get_unreviewed_faces(session)
    
    if not unknowns:
        st.success("All clear! No pending unknown faces.")
        return

    st.warning(f"You have {len(unknowns)} faces pending review.")
    
    for u in unknowns:
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 2, 1])
            # Image placeholder (if physical file exists)
            if u.image_path and os.path.exists(u.image_path):
                c1.image(u.image_path, width=150)
            else:
                c1.write("🖼️ (Image missing)")
                
            c2.write(f"**Detected at:** {u.timestamp}")
            c2.write(f"**Peak Confidence:** {u.confidence:.2f}")
            
            if c3.button(f"Review Done #{u.id}"):
                db_manager.mark_unknown_face_reviewed(session, u.id)
                st.success("Marked as reviewed")
                st.rerun()

def render_system_health(session):
    st.header("⚙️ System Health & Configuration")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Hardware & GPU")
        gpu_ready = gpu_mgr.is_gpu_ready()
        st.success("NVIDIA GPU Ready" if gpu_ready else "Running on CPU (GPU not found)")
        st.write("**Model:** NVIDIA GeForce GTX 1050")
        st.write("**Acceleration:** CUDA 12.1 + cuDNN 9.1")
        st.write("**Detector:** RetinaFace (ONNX Optimized)")
        st.write("**Recognizer:** Facenet512 + ArcFace")
        
    with c2:
        st.subheader("Storage & Filesystem")
        if os.path.exists(UNKNOWN_FACES_DIR):
            log_files = list(Path(UNKNOWN_FACES_DIR).glob("*.jpg"))
            st.write(f"**Captured Unknown Faces:** {len(log_files)}")
            # Simple size calculation
            total_size = sum(f.stat().st_size for f in log_files) / (1024 * 1024)
            st.write(f"**Storage Used:** {total_size:.2f} MB")
        
        st.write(f"**Database:** {DATABASE_URL.split('/')[-1]}")

    st.divider()
    st.subheader("Recent System Logs")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()[-20:]
            st.code("".join(lines))
    else:
        st.info("System log file not found.")


# --- Main App ---

def main():
    st.set_page_config(
        page_title="EMSVIA Admin Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for premium look
    st.markdown("""
        <style>
        .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #d1d5db; }
        .stActionButton { background-color: #4CAF50; color: white; }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("💠 EMSVIA Portal")
    st.sidebar.image("https://img.icons8.com/isometric/512/facial-recognition.png", width=80)
    
    menu = ["Dashboard Overview", "Student Directory", "Attendance History", "Unknown Face Queue", "System Health"]
    choice = st.sidebar.radio("Navigate System", menu)
    
    st.sidebar.divider()
    st.sidebar.info("Running on NVIDIA GTX 1050\n(WSL2 Environment)")

    session = db_manager.get_session()
    try:
        if choice == "Dashboard Overview":
            render_overview(session)
        elif choice == "Student Directory":
            render_students(session)
        elif choice == "Attendance History":
            render_attendance(session)
        elif choice == "Unknown Face Queue":
            render_unknowns(session)
        elif choice == "System Health":
            render_system_health(session)
    finally:
        session.close()


if __name__ == "__main__":
    main()
