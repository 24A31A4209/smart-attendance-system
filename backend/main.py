import os
import cv2
import numpy as np
import base64
import csv
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, send_file
from insightface.app import FaceAnalysis
# --- NEW DATABASE IMPORTS ---
from data import init_db, save_student_face, get_known_faces, save_daily_attendance # Added save_daily_attendance
import sqlite3 # Needed for the history query

app = Flask(__name__)

# --- PATH LOGIC ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
BASE_DB_PATH = os.path.join(BASE_DIR, "database", "branch_data")

if not os.path.exists(BASE_DB_PATH):
    os.makedirs(BASE_DB_PATH, exist_ok=True)

# --- AI INITIALIZATION ---
face_app = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640)) 

@app.route('/')
def index():
    return render_template('index.html')

# --- 1. ENROLLMENT (UPDATED TO USE SQLITE) ---
@app.route('/enroll_capture', methods=['POST'])
def enroll_capture():
    try:
        data = request.json
        roll_no = data.get('roll_no')
        branch_section = data.get('branch_section')
        image_data = data.get('image').split(',')[1]

        img_bytes = base64.decodebytes(image_data.encode())
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = face_app.get(rgb_img)
        if not faces:
            return jsonify({"success": False, "error": "AI could not see your face."})

        # Keep saving the image for your records
        user_dir = os.path.join(BASE_DB_PATH, branch_section, roll_no)
        os.makedirs(user_dir, exist_ok=True)
        cv2.imwrite(os.path.join(user_dir, "enrolled_face.jpg"), img)
        
        # Save to Database instead of Pickle
        embedding = faces[0].normed_embedding
        success = save_student_face(roll_no, branch_section, embedding)

        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Database error"})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# --- 2. PHOTO PROCESSING (UPDATED TO USE SQLITE) ---
@app.route('/process_photo', methods=['POST'])
def process_photo():
    try:
        branch_section = request.form.get('branch_section')
        file = request.files['class_photo']
        
        # Pull faces from Database
        known_faces = get_known_faces(branch_section)
        if not known_faces:
            return jsonify({"error": "No students enrolled in this branch"}), 404

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        h, w, _ = img.shape
        img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Your original Tiling Logic
        h_step, w_step, overlap = h // 2, w // 3, 150 
        tiles = []
        for i in range(2):
            for j in range(3):
                y1, y2 = max(0, i * h_step - overlap), min(h, (i + 1) * h_step + overlap)
                x1, x2 = max(0, j * w_step - overlap), min(w, (j + 1) * w_step + overlap)
                tiles.append(rgb_img[y1:y2, x1:x2])

        present = set()
        face_app.prepare(ctx_id=0, det_size=(640, 640)) 

        for tile in tiles:
            if tile.size == 0: continue
            detected_faces = face_app.get(tile)
            for face in detected_faces:
                if face.det_score < 0.10: continue 
                
                best_score, best_roll = -1, None
                for roll, known_vec in known_faces.items():
                    score = np.dot(face.normed_embedding, known_vec)
                    if score > best_score:
                        best_score, best_roll = score, roll
                
                if best_score > 0.23:
                    present.add(best_roll)

        absent = list(set(known_faces.keys()) - present)

        return jsonify({
            "present_count": len(present),
            "present_list": sorted(list(present)),
            "absent_count": len(absent),
            "absent_list": sorted(absent),
            "message": f"Detected {len(present)} students"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 3. VIDEO PROCESSING (UPDATED TO USE SQLITE) ---
@app.route('/process_video', methods=['POST'])
def process_video():
    try:
        branch_section = request.form.get('branch_section')
        video_file = request.files['class_video']
        temp_path = "temp_video.mp4"
        video_file.save(temp_path)
        
        known_faces = get_known_faces(branch_section)
        if not known_faces: return jsonify({"error": "DB empty"}), 404
            
        cap = cv2.VideoCapture(temp_path)
        present = set()
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if count % int(fps) == 0: 
                small_rgb = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB)
                faces = face_app.get(small_rgb)
                for face in faces:
                    for roll, known_vec in known_faces.items():
                        if np.dot(face.normed_embedding, known_vec) > 0.38:
                            present.add(roll)
            count += 1
            
        cap.release()
        if os.path.exists(temp_path): os.remove(temp_path)
        absent = list(set(known_faces.keys()) - present)
        
        return jsonify({
            "present_count": len(present), "present_list": sorted(list(present)), 
            "absent_count": len(absent), "absent_list": sorted(absent)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 4. EXPORT ATTENDANCE ---
@app.route('/export_attendance', methods=['POST'])
def export_attendance():
    try:
        data = request.json
        branch_section = data.get('branch_section', 'Class_Report')
        present_list = data.get('present_list', [])
        absent_list = data.get('absent_list', [])
        
        # --- NEW: SAVE TO DATABASE HISTORY ---
        save_daily_attendance(present_list, absent_list, branch_section)
        # -------------------------------------

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Attendance_{branch_section}_{timestamp}.csv"
        filepath = os.path.join(BASE_DIR, filename)

        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Smart Attendance Report"])
            writer.writerow(["Branch/Section", branch_section])
            writer.writerow(["Date & Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow([])
            writer.writerow(["Roll Number", "Status"])
            for roll in present_list: writer.writerow([roll, "PRESENT"])
            for roll in absent_list: writer.writerow([roll, "ABSENT"])

        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
#=========== LIVE SESSION LOGIC (UPDATED TO USE SQLITE) ============

#=========== REFINED LIVE SESSION LOGIC ============

# Global to track attendance during the session
active_session = {"present": set(), "is_running": False}

@app.route('/start_live_session/<bs>')
def start_live_session(bs):
    active_session["present"] = set()
    active_session["is_running"] = True
    return jsonify({"status": "session_started", "branch": bs})

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    # Only process if a session is actually active
    if not active_session["is_running"]:
        return jsonify({"success": False, "error": "Session not active"})

    try:
        data = request.json
        branch_section = data.get('branch_section')
        # Check if image data exists to prevent split errors
        if not data.get('image'):
            return jsonify({"success": False, "error": "No image data"})
            
        image_data = data.get('image').split(',')[1]

        # Decode & Process
        img_bytes = base64.b64decode(image_data) # Use b64decode for standard base64
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        known_faces = get_known_faces(branch_section)
        if not known_faces:
             return jsonify({"success": True, "results": [], "message": "No enrolled students"})

        # Run detection
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_app.get(rgb_img)
        
        results = []
        for face in faces:
            best_roll = "Unknown"
            max_sim = 0
            
            # Recognition Logic
            for roll, vec in known_faces.items():
                sim = np.dot(face.normed_embedding, vec)
                if sim > 0.45: # Threshold
                    if sim > max_sim:
                        max_sim = sim
                        best_roll = roll
            
            if best_roll != "Unknown":
                active_session["present"].add(best_roll)
            
            # Send BBOX as list for JSON compatibility
                bbox = [int(x) for x in face.bbox.tolist()]
                results.append({"name": best_roll, "bbox": bbox})
        return jsonify({"success": True, "results": results})
    except Exception as e:
        print(f"Error: {e}") # Log error to console for debugging
        return jsonify({"success": False, "error": str(e)})

@app.route('/stop_live_session/<bs>')
def stop_live_session(bs):
    active_session["is_running"] = False
    known_faces = get_known_faces(bs)
    
    p_list = sorted(list(active_session["present"]))
    # Calculate absent students based on the full class list
    all_students = set(known_faces.keys())
    a_list = sorted(list(all_students - active_session["present"]))
    
    return jsonify({
        "present_count": len(p_list), 
        "present_list": p_list, 
        "absent_count": len(a_list), 
        "absent_list": a_list
    })
#=========== HISTORY VIEWER LOGIC (NEW) ============
@app.route('/history')
def history():
    selected_branch = request.args.get('branch')
    selected_date = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
    history_records = []

    if selected_branch and selected_date:
        # Connect to DB to fetch the requested records
        from data import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT roll_no, status FROM attendance_history 
            WHERE branch_section = ? AND date = ? 
            ORDER BY roll_no ASC
        ''', (selected_branch, selected_date))
        history_records = cursor.fetchall()
        conn.close()

    return render_template('history.html', 
                           history=history_records, 
                           selected_branch=selected_branch, 
                           selected_date=selected_date)

#=========== EXPORT HISTORY AS CSV (NEW) ============

@app.route('/export_history_csv')
def export_history_csv():
    branch = request.args.get('branch')
    date = request.args.get('date')
    
    if not branch or not date:
        return "Missing parameters", 400

    from data import DB_PATH
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT roll_no, status FROM attendance_history 
        WHERE branch_section = ? AND date = ? 
        ORDER BY roll_no ASC
    ''', (branch, date))
    rows = cursor.fetchall()
    conn.close()

    # Create the CSV in memory
    filename = f"History_{branch}_{date}.csv"
    filepath = os.path.join(BASE_DIR, filename)

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Attendance History Report"])
        writer.writerow(["Branch", branch])
        writer.writerow(["Date", date])
        writer.writerow([])
        writer.writerow(["Roll Number", "Status"])
        for roll, status in rows:
            writer.writerow([roll, status])

    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    init_db() # Initializes the SQLite Database
    app.run(debug=True, port=5000)