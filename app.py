from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
from flask_cors import CORS
from datetime import datetime
import os
import sqlite3
from detection import AccidentDetectionModel
from email_sender import send_alert_email
from pathlib import Path

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for debugging

# Load your trained model
model = AccidentDetectionModel("model.json", 'model_weights.h5')


def init_db():
    conn = sqlite3.connect('accidents.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS accidents
                 (id TEXT, date TEXT, address TEXT, severity INTEGER, severityLabel TEXT, image_url TEXT)''')
    conn.commit()
    conn.close()

init_db()


# Dummy user database (replace with a real database in production)
users = {
    'nguyenphuccuongtm@gmail.com': 'Cuongbk2003'
}

# Serve the index.html file for the root URL
@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

# Serve static files (e.g., favicon.ico, if needed)
@app.route('/favicon.ico')
def serve_favicon():
    return '', 204  # Return empty response with 204 (No Content) status

# Serve image files from the static/images directory
@app.route('/static/images/<filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)

# Login endpoint
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400

        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password are required'}), 400

        if email in users and users[email] == password:
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error during login: {str(e)}'}), 500

@app.route('/accidents', methods=['GET'])
def get_accidents():
    conn = sqlite3.connect('accidents.db')
    c = conn.cursor()
    c.execute("SELECT * FROM accidents")
    rows = c.fetchall()
    conn.close()
    return jsonify([{'id': row[0], 'date': row[1], 'address': row[2], 'severity': row[3], 'severityLabel': row[4], 'image_url': row[5]} for row in rows])

def ensure_serializable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return None  # Exclude NumPy arrays
    elif isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_serializable(item) for item in obj]
    return obj


def process_video(video_path):
    print(f"Processing video: {video_path}")
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return {
                'accident': False,
                'severity': 0.0,
                'location': "Unknown",
                'error': "Could not open video file"
            }

        frames = []  # Frames for prediction (resized and normalized)
        original_frames = []  # Original frames for saving the best one
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            # Keep the original frame for saving later
            original_frames.append(frame.copy())
            # Preprocess frame for prediction
            frame = cv2.resize(frame, (250, 250))  # Adjust size based on your model
            # frame = frame / 255.0  # Normalize
            frames.append(frame)
        
        cap.release()
        print(f"Extracted {frame_count} frames")
        
        # Check if frames are empty
        if len(frames) == 0:
            print("Error: No frames extracted from video")
            return {
                'accident': False,
                'severity': 0.0,
                'location': "Unknown",
                'error': "No frames extracted from video"
            }
        
        # Convert frames to numpy array for prediction
        frames = np.array(frames)
        print(f"Frames shape: {frames.shape}")
        
        # Predict using the model
        if model is None:
            print("Error: Model not loaded")
            return {
                'accident': False,
                'severity': 0.0,
                'location': "Unknown",
                'error': "Model not loaded"
            }
        
        # Predict on all frames
        accident_scores = []
        predictions = []
        for i, frame in enumerate(frames):
            pred, prob = model.predict_accident(frame[np.newaxis, :, :])
            print(f"Frame {i} - Prediction: {pred}, Probability: {prob}")
            
            predictions.append(pred)
            # Handle different possible formats of prob
            if isinstance(prob, (list, np.ndarray)):
                # Assuming prob is like [no_accident_prob, accident_prob] or [[no_accident_prob, accident_prob]]
                prob_array = np.array(prob).flatten()
                
                accident_prob = float(prob_array[0])
            else:
                # If prob is a scalar
                accident_prob = float(prob)
            if accident_prob < 0.98:
                accident_scores.append(accident_prob)
        
        accident_scores = np.array(accident_scores)
        print(f"Accident scores: {accident_scores}")

        # Process predictions
        location = "298, Cầu Diễn, Minh Khai, Bắc Từ Liêm, Hà Nội"  # Dummy location
        max_score = np.max(accident_scores) if len(accident_scores) > 0 else 0.0
        accident_detected = max_score > 0.9  # Threshold for accident detection
        severity = max_score * 100  # Convert probability to percentage

        # If an accident is detected, find the frame with the highest score
        if accident_detected:
            max_pred_index = np.argmax(accident_scores)
            best_frame = original_frames[max_pred_index]
            print(f"Frame with highest accident detection score: {max_pred_index}, Score: {float(accident_scores[max_pred_index])}")
            return {
                'accident': predictions[max_pred_index],  # "Accident" or "No Accident"
                'severity': round(severity, 2),
                'location': location,
                'best_frame': best_frame
            }
        else:
            return {
                'accident': "No Accident",
                'severity': round(severity, 2),
                'location': location
            }
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return {
            'accident': False,
            'severity': 0.0,
            'location': "Unknown",
            'error': f"Error processing video: {str(e)}"
        }
    

@app.route('/predict', methods=['POST'])
def predict():
    print("Received request to /predict")
    if 'video' not in request.files:
        print("Error: No video uploaded")
        return jsonify({'error': 'No video uploaded'}), 400
    
    video = request.files['video']
    print(f"Video file received: {video.filename}, Size: {video.content_length}")
    
    # Save the video temporarily to process it
    temp_video_path = os.path.join('static/temp', f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4")
    os.makedirs(os.path.dirname(temp_video_path), exist_ok=True)
    try:
        video.save(temp_video_path)
        print(f"Temporary video saved to {temp_video_path}")
    except Exception as e:
        print(f"Error saving temporary video: {str(e)}")
        return jsonify({'error': f"Failed to save video: {str(e)}"}), 500
    
    # Process the video
    result = process_video(temp_video_path)
    os.remove(temp_video_path)  # Clean up temporary video
    if 'error' in result:
        print(f"Returning error: {result['error']}")
        return jsonify({'error': result['error']}), 500
    
    # If an accident is detected, save to database
    if result['accident']:
        conn = sqlite3.connect('accidents.db')
        c = conn.cursor()

        image_filename = f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        image_path = os.path.join('static/images', image_filename)

        try:
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
        except Exception as e:
            print(f"Error creating directory for image: {str(e)}")
            return jsonify({'error': f"Failed to create directory for image: {str(e)}"}), 500
        
        try:
            cv2.imwrite(image_path, result['best_frame'])
            print(f"Best frame saved to {image_path}")
        except Exception as e:
            print(f"Error saving best frame: {str(e)}")
            return jsonify({'error': f"Failed to save best frame: {str(e)}"}), 500

        date = datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT')
        # image_path = Path("D:/HAUI/DATN/Accident_detection/Code1/Accident-Detection-System/app/static/images") / image_filename

        # new_accident = (
        #     f"acc-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        #     date,
        #     result['location'],
        #     result['severity'],
        #     'Nghiêm trọng' if result['severity'] > 70 else 'Trung bình' if result['severity'] > 50 else 'Thấp',
        #     image_path
        # )

        # c.execute("INSERT INTO accidents (id, date, address, severity, severityLabel, image_url) VALUES (?, ?, ?, ?, ?, ?)", new_accident)
        # conn.commit()
        # conn.close()
        # print(f"Accident added to database: {new_accident}")

        # send_alert_email(date, result['location'], image_path)

        # Define the relative path for the database
        image_url = f"/static/images/{image_filename}"
        # Define the absolute path for saving the image
        image_path = os.path.join('static/images', image_filename)

        try:
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
        except Exception as e:
            print(f"Error creating directory for image: {str(e)}")
            return jsonify({'error': f"Failed to create directory for image: {str(e)}"}), 500

        try:
            cv2.imwrite(image_path, result['best_frame'])
            print(f"Best frame saved to {image_path}")
        except Exception as e:
            print(f"Error saving best frame: {str(e)}")
            return jsonify({'error': f"Failed to save best frame: {str(e)}"}), 500

        date = datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT')

        new_accident = (
            f"acc-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            date,
            result['location'],
            result['severity'],
            'Nghiêm trọng' if result['severity'] > 70 else 'Trung bình' if result['severity'] > 50 else 'Thấp',
            image_url  # Use the relative path for the database
        )

        c.execute("INSERT INTO accidents (id, date, address, severity, severityLabel, image_url) VALUES (?, ?, ?, ?, ?, ?)", new_accident)
        conn.commit()
        conn.close()
        print(f"Accident added to database: {new_accident}")

        # Pass the absolute path to send_alert_email
        absolute_image_path = os.path.abspath(image_path)
        send_alert_email(date, result['location'], absolute_image_path)
    
    # Remove the best_frame from the result before sending to the client
    result_to_send = ensure_serializable(result)
    result_to_send = {k: v for k, v in result_to_send.items() if k != 'best_frame'}
    print(f"Returning result: {result_to_send}")
    return jsonify(result_to_send)


@app.route('/accident-stats', methods=['GET'])
def get_accident_stats():
    try:
        print("Connecting to database...")
        conn = sqlite3.connect('accidents.db')
        c = conn.cursor()
        print("Executing query to count accidents by month...")
        # Fetch the raw data
        c.execute("""
            SELECT 
                substr(date, 13, 4) || '-' ||
                (CASE 
                    WHEN substr(date, 9, 3) = 'Jan' THEN '01'
                    WHEN substr(date, 9, 3) = 'Feb' THEN '02'
                    WHEN substr(date, 9, 3) = 'Mar' THEN '03'
                    WHEN substr(date, 9, 3) = 'Apr' THEN '04'
                    WHEN substr(date, 9, 3) = 'May' THEN '05'
                    WHEN substr(date, 9, 3) = 'Jun' THEN '06'
                    WHEN substr(date, 9, 3) = 'Jul' THEN '07'
                    WHEN substr(date, 9, 3) = 'Aug' THEN '08'
                    WHEN substr(date, 9, 3) = 'Sep' THEN '09'
                    WHEN substr(date, 9, 3) = 'Oct' THEN '10'
                    WHEN substr(date, 9, 3) = 'Nov' THEN '11'
                    WHEN substr(date, 9, 3) = 'Dec' THEN '12'
                    ELSE '00'
                END) AS month,
                COUNT(*)
            FROM accidents
            GROUP BY month
            ORDER BY month
        """)
        rows = c.fetchall()
        print(f"Query returned {len(rows)} rows: {rows}")
        conn.close()

        # Define all 12 months for the year 2025
        year = "2025"  # You can dynamically determine the year if needed
        month_names = [
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ]
        month_numbers = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        labels = [f"{month}-{year}" for month in month_numbers]
        data = [0] * 12  # Initialize counts to 0 for all months

        # Fill in the counts from the database
        for row in rows:
            if row[0] is None:
                continue
            # row[0] is in "YYYY-MM" format (e.g., "2025-04")
            row_year, row_month = row[0].split('-')
            if row_year == year:
                # Map the month number to the index (e.g., "04" -> index 3)
                month_index = month_numbers.index(row_month)
                data[month_index] = row[1]  # Set the count for that month

        print(f"Returning chart data: labels={labels}, data={data}")

        return jsonify({
            'labels': labels,
            'data': data
        })
    except Exception as e:
        print(f"Error fetching accident stats: {str(e)}")
        return jsonify({'error': f"Error fetching accident stats: {str(e)}"}), 500


@app.route('/accidents/<id>', methods=['DELETE'])
def delete_accident(id):
    try:
        conn = sqlite3.connect('accidents.db')
        c = conn.cursor()
        # Check if the accident exists
        c.execute("SELECT * FROM accidents WHERE id = ?", (id,))
        accident = c.fetchone()
        if not accident:
            conn.close()
            return jsonify({'error': 'Accident not found'}), 404

        # Delete the accident
        c.execute("DELETE FROM accidents WHERE id = ?", (id,))
        conn.commit()
        
        # Optionally, delete the associated image file
        image_path = accident[5]  # image_url is the 6th column (index 5)
        if image_path and os.path.exists(image_path.lstrip('/')):
            try:
                if image_path and os.path.exists(image_path.lstrip('/')):
                    os.remove(image_path.lstrip('/'))
                    print(f"Deleted image file: {image_path}")
            except Exception as e:
                print(f"Error deleting image file: {str(e)}")

        conn.close()
        print(f"Deleted accident with ID: {id}")
        return jsonify({'success': True, 'message': 'Accident deleted successfully'})
    except Exception as e:
        print(f"Error deleting accident: {str(e)}")
        return jsonify({'error': f"Error deleting accident: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)