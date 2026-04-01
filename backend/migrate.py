import os
import pickle
import numpy as np
from data import init_db, save_student_face

# Define the path where your old .pkl files are stored
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OLD_DB_PATH = os.path.join(BASE_DIR, "database", "branch_data")

def run_migration():
    print("--- Starting Migration ---")
    
    # 1. Initialize the new SQLite database
    init_db()
    
    if not os.path.exists(OLD_DB_PATH):
        print(f"Error: Could not find old data at {OLD_DB_PATH}")
        return

    # 2. Loop through every branch folder (e.g., CSE_A, ECE_B)
    branches = [d for d in os.listdir(OLD_DB_PATH) if os.path.isdir(os.path.join(OLD_DB_PATH, d))]
    
    total_migrated = 0

    for branch in branches:
        pkl_path = os.path.join(OLD_DB_PATH, branch, f"{branch}_db.pkl")
        
        if os.path.exists(pkl_path):
            print(f"\nProcessing branch: {branch}")
            
            try:
                # 3. Open the old pickle file
                with open(pkl_path, 'rb') as f:
                    old_data = pickle.load(f) # Format: {roll_no: embedding_vector}
                
                # 4. Insert each student into the new SQL database
                for roll_no, embedding in old_data.items():
                    success = save_student_face(roll_no, branch, embedding)
                    if success:
                        print(f"  [OK] Migrated: {roll_no}")
                        total_migrated += 1
                    else:
                        print(f"  [FAILED] Could not migrate: {roll_no}")
            
            except Exception as e:
                print(f"  [ERROR] Problem reading {pkl_path}: {e}")
        else:
            print(f"\nSkipping {branch}: No .pkl file found.")

    print(f"\n--- Migration Finished! ---")
    print(f"Total students moved to SQL: {total_migrated}")

if __name__ == "__main__":
    run_migration()