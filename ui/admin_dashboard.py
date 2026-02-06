import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.settings import DATABASE_URL
from database.crud import AttendanceDB

# Initialize DB
db_manager = AttendanceDB(DATABASE_URL)

def main():
    st.set_page_config(page_title="Attendance System Admin Dashboard", layout="wide")
    
    st.title("📊 Face Recognition Attendance System - Admin Dashboard")
    st.sidebar.title("Navigation")
    
    page = st.sidebar.selectbox("Choose a page", ["Overview", "Student Management", "Attendance History"])
    
    session = db_manager.get_session()
    
    try:
        if page == "Overview":
            st.header("System Overview")
            
            students = db_manager.get_all_students(session)
            attendance = db_manager.get_attendance_history(session)
            unknowns = db_manager.get_unreviewed_faces(session)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Enrolled Students", len(students))
            col2.metric("Total Attendance Logs", len(attendance))
            col3.metric("Unreviewed Unknowns", len(unknowns))
            
            st.subheader("Recent Attendance")
            if attendance:
                df = pd.DataFrame([{
                    "Timestamp": r.timestamp,
                    "Student ID": r.student_id,
                    "Status": r.status,
                    "Confidence": r.confidence_score
                } for r in attendance[:10]])
                st.table(df)
            else:
                st.info("No attendance records found.")

        elif page == "Student Management":
            st.header("Student Management")
            students = db_manager.get_all_students(session)
            
            if students:
                df = pd.DataFrame([{
                    "DB ID": s.id,
                    "Student ID": s.student_id,
                    "First Name": s.first_name,
                    "Last Name": s.last_name,
                    "Email": s.email,
                    "Enrolled On": s.enrollment_date
                } for s in students])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No students enrolled yet.")

        elif page == "Attendance History":
            st.header("Attendance History")
            history = db_manager.get_attendance_history(session)
            
            if history:
                df = pd.DataFrame([{
                    "Timestamp": h.timestamp,
                    "Student ID": h.student_id,
                    "Status": h.status,
                    "Confidence": h.confidence_score,
                    "Camera": h.camera_id
                } for h in history])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No attendance records found.")

    finally:
        session.close()

if __name__ == "__main__":
    main()
