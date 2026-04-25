import sqlite3
import os

db_path = "database/attendance.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE attendance_records ADD COLUMN entry_type VARCHAR(20) DEFAULT 'entry';")
        conn.commit()
        print("Successfully added entry_type column to attendance_records")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("Column entry_type already exists.")
        else:
            print(f"Error: {e}")
    finally:
        conn.close()
else:
    print(f"Database {db_path} not found.")
