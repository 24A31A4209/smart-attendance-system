import sqlite3
import numpy as np
import os
from datetime import datetime

# Path to the database file
DB_PATH = os.path.join(os.path.dirname(__file__), "database", "attendance.db")

def init_db():
    """Creates the database and tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Table for Student Face Data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_no TEXT UNIQUE NOT NULL,
            branch_section TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    ''')

    # 2. NEW: Table for Attendance History
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_no TEXT NOT NULL,
            branch_section TEXT NOT NULL,
            date TEXT NOT NULL,
            status TEXT NOT NULL,
            UNIQUE(roll_no, date) 
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database & History Tables Initialized.")

def save_student_face(roll_no, branch_section, embedding):
    """Saves or updates a student's face embedding."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    binary_embedding = embedding.astype(np.float32).tobytes()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO students (roll_no, branch_section, embedding)
            VALUES (?, ?, ?)
        ''', (roll_no, branch_section, binary_embedding))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving student: {e}")
        return False
    finally:
        conn.close()

def save_daily_attendance(present_list, absent_list, branch_section):
    """Saves the daily status for all students in a branch."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # Insert Present Students
        for roll in present_list:
            cursor.execute('''
                INSERT OR REPLACE INTO attendance_history (roll_no, branch_section, date, status)
                VALUES (?, ?, ?, 'PRESENT')
            ''', (roll, branch_section, today))
            
        # Insert Absent Students
        for roll in absent_list:
            cursor.execute('''
                INSERT OR REPLACE INTO attendance_history (roll_no, branch_section, date, status)
                VALUES (?, ?, ?, 'ABSENT')
            ''', (roll, branch_section, today))
            
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving history: {e}")
        return False
    finally:
        conn.close()

def get_known_faces(branch_section):
    """Retrieves all faces for a specific branch."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Adding ORDER BY roll_no here keeps your database organized when loading
    cursor.execute('SELECT roll_no, embedding FROM students WHERE branch_section = ? ORDER BY roll_no ASC', (branch_section,))
    rows = cursor.fetchall()
    conn.close()

    known_faces = {}
    for roll_no, binary_blob in rows:
        known_faces[roll_no] = np.frombuffer(binary_blob, dtype=np.float32)
    
    return known_faces