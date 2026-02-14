from database.crud import AttendanceDB
from config.settings import DATABASE_URL
from database.models import Student

db = AttendanceDB(DATABASE_URL)
session = db.get_session()

students = session.query(Student).all()
print(f"Check: Found {len(students)} students.")
for s in students:
    print(f"PK: {s.id}, ID: {s.student_id}, Name: {s.first_name} {s.last_name}")

session.close()
