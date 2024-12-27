import os
import json
from datetime import datetime


def check_attendance(name):
    # Define the base directory for attendance files
    base_dir = os.path.join(os.getcwd(), ".attendance")
    os.makedirs(base_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Get current date and construct the filename
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join(base_dir, f"{current_date}.json")

    # Prepare attendance entry
    attendance_entry = {
        "name": name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Load existing attendance or create a new file
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            attendance_data = json.load(file)
    else:
        attendance_data = []

    # Append the new attendance entry
    attendance_data.append(attendance_entry)

    # Write back to the file
    with open(file_path, "w") as file:
        json.dump(attendance_data, file, indent=4)

    print(f"Attendance recorded for {name} at {attendance_entry['timestamp']}")
