from flask import Flask, render_template, request, Response, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
import threading
import uuid
from pathlib import Path
import queue
import csv
import datetime
import io

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = '/tmp/uploads'  # Use /tmp for Vercel
RESULT_FOLDER = '/tmp/results'   # Use /tmp for Vercel
CSV_FOLDER = '/tmp/csv'          # Use /tmp for Vercel
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, RESULT_FOLDER, CSV_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['CSV_FOLDER'] = CSV_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

# Global state
model = None  # We'll load the model on-demand instead of at startup
processing_jobs = {}  # Track processing jobs
stream_queues = {}  # Store frames for live streaming

def get_model():
    """Get or initialize the YOLO model"""
    global model
    if model is None:
        model = YOLO("best.pt")
    return model

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path, result_path, csv_path=None):
    """Process a single image with YOLO model and save the result"""
    try:
        # Read the image
        image = cv2.imread(image_path)
        
        # Get model instance
        model = get_model()
        
        # Run inference
        results = model(image)
        
        # Get the annotated frame with detections
        annotated_frame = results[0].plot()
        
        # Process detection results
        detections = {}
        recent_detections = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # Get detection classes, confidence scores, and boxes
            detected_classes = set()  # Track unique class instances
            
            for i, box in enumerate(results[0].boxes):
                cls = int(box.cls[0].item())
                conf = box.conf[0].item()
                class_name = results[0].names[cls]
                
                # Add to count (only once per unique class)
                if class_name not in detected_classes:
                    detected_classes.add(class_name)
                    if class_name in detections:
                        detections[class_name] += 1
                    else:
                        detections[class_name] = 1
                
                # Add to recent detections
                detection_info = {
                    'class': class_name,
                    'confidence': round(conf * 100, 2),
                    'position': f"({int(box.xyxy[0][0])}, {int(box.xyxy[0][1])})"
                }
                recent_detections.append(detection_info)
                
        # Save the result
        cv2.imwrite(result_path, annotated_frame)
        
        # Save to CSV if requested
        if csv_path:
            save_detections_to_csv(detections, recent_detections, csv_path, is_video=False)
        
        return True, detections, recent_detections
    except Exception as e:
        print(f"Error processing image: {e}")
        return False, {}, []

def process_video_with_stream(video_path, result_path, job_id, csv_path=None):
    """Process a video with YOLO model and save the result, while also streaming the processed frames"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            processing_jobs[job_id]['status'] = 'failed'
            return False
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
        
        # Create a queue for this job
        stream_queues[job_id] = queue.Queue(maxsize=30)  # Limit queue size
        
        # Get model instance
        model = get_model()
        
        frame_count = 0
        
        # Read and process video frame-by-frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run inference
            results = model(frame)
            
            # Get the annotated frame with detections
            annotated_frame = results[0].plot()
            
            # Process detection results
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                # Get detection classes, confidence scores, and boxes
                for i, box in enumerate(results[0].boxes):
                    cls = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    class_name = results[0].names[cls]
                    
                    # Only count unique class instances (not duplicates across frames)
                    if class_name not in processing_jobs[job_id]['unique_classes']:
                        processing_jobs[job_id]['unique_classes'].add(class_name)
                        if class_name in processing_jobs[job_id]['detections']:
                            processing_jobs[job_id]['detections'][class_name] += 1
                        else:
                            processing_jobs[job_id]['detections'][class_name] = 1
                    
                    # Add to recent detections (limit to last 5)
                    processing_jobs[job_id]['recent_detections'].append({
                        'class': class_name,
                        'confidence': round(conf * 100, 2),
                        'frame': frame_count
                    })
                    
                    # Keep only the last 5 detections
                    processing_jobs[job_id]['recent_detections'] = processing_jobs[job_id]['recent_detections'][-5:]
            
            # Write the frame to the output video
            out.write(annotated_frame)
            
            # Add the frame to the streaming queue
            # Convert to JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if ret:
                # If queue is full, remove oldest frame
                if stream_queues[job_id].full():
                    try:
                        stream_queues[job_id].get_nowait()
                    except queue.Empty:
                        pass
                # Add new frame
                stream_queues[job_id].put(buffer.tobytes())
            
            # Update job progress
            frame_count += 1
            progress = int(100 * frame_count / total_frames)
            processing_jobs[job_id]['progress'] = progress
            processing_jobs[job_id]['frame_count'] = frame_count
            
        # Release resources
        cap.release()
        out.release()
        
        # Update job status
        processing_jobs[job_id]['status'] = 'completed'
        processing_jobs[job_id]['result_path'] = os.path.basename(result_path)
        
        # Save to CSV if requested
        if csv_path:
            save_detections_to_csv(
                processing_jobs[job_id]['detections'], 
                processing_jobs[job_id]['recent_detections'], 
                csv_path, 
                is_video=True
            )
        
        # Clean up queue
        if job_id in stream_queues:
            del stream_queues[job_id]
        
        return True
    except Exception as e:
        print(f"Error processing video: {e}")
        processing_jobs[job_id]['status'] = 'failed'
        processing_jobs[job_id]['error'] = str(e)
        return False

def save_detections_to_csv(detections, detailed_detections, csv_path, is_video=False):
    """Save detection results to CSV file"""
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            # CSV writer
            writer = csv.writer(csvfile)
            
            # Write header and timestamp
            writer.writerow(['Detection Results'])
            writer.writerow(['Generated on', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow([])
            
            # Write summary section
            writer.writerow(['Object Summary'])
            writer.writerow(['Class', 'Count'])
            
            for class_name, count in detections.items():
                writer.writerow([class_name, count])
            
            writer.writerow([])
            
            # Write detailed detections
            writer.writerow(['Detailed Detections'])
            
            if is_video:
                writer.writerow(['Class', 'Confidence (%)', 'Frame'])
            else:
                writer.writerow(['Class', 'Confidence (%)', 'Position'])
            
            for detection in detailed_detections:
                if is_video:
                    writer.writerow([detection['class'], detection['confidence'], detection['frame']])
                else:
                    writer.writerow([detection['class'], detection['confidence'], detection['position']])
                    
        return True
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return False

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/detection')
def detection():
    """Render the detection page"""
    return render_template('detection.html')

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        # Generate unique ID for this job
        job_id = str(uuid.uuid4())
        
        # Save the file
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(upload_path)
        
        # Determine if it's an image or video
        is_video = file_extension in {'mp4', 'avi', 'mov'}
        
        # Create result path
        if is_video:
            result_filename = f"{job_id}_result.mp4"
        else:
            result_filename = f"{job_id}_result.jpg"
            
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        
        # Create CSV path
        csv_filename = f"{job_id}_detections.csv"
        csv_path = os.path.join(app.config['CSV_FOLDER'], csv_filename)
        
        # Process based on file type
        if is_video:
            # Create initial job entry with tracking for unique object classes
            processing_jobs[job_id] = {
                'status': 'processing',
                'progress': 0,
                'file_type': 'video',
                'original_filename': filename,
                'detections': {},
                'recent_detections': [],
                'unique_classes': set(),  # Track unique classes to avoid duplicate counts
                'csv_path': csv_path
            }
            
            # Start processing in a separate thread and return job ID
            thread = threading.Thread(
                target=process_video_with_stream, 
                args=(upload_path, result_path, job_id, csv_path)
            )
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'job_id': job_id,
                'status': 'processing'
            })
        else:
            # For images, process immediately
            success, detections, recent_detections = process_image(upload_path, result_path, csv_path)
            
            if success:
                return jsonify({
                    'status': 'completed',
                    'result_path': f"/tmp/results/{result_filename}",  # Updated path
                    'csv_path': f"/tmp/csv/{csv_filename}",  # Updated path
                    'file_type': 'image',
                    'detections': detections,
                    'recent_detections': recent_detections
                })
            else:
                return jsonify({'error': 'Failed to process image'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get the status of a processing job"""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
        
    job = processing_jobs[job_id]
    
    response = {
        'status': job['status'],
        'progress': job.get('progress', 100)
    }
    
    # Add detection data if available
    if 'detections' in job:
        response['detections'] = job['detections']
    
    if 'recent_detections' in job:
        response['recent_detections'] = job['recent_detections']
    
    if 'frame_count' in job:
        response['frame_count'] = job['frame_count']
    
    if job['status'] == 'completed':
        response['result_path'] = f"/tmp/results/{job['result_path']}"  # Updated path
        response['csv_path'] = f"/tmp/csv/{os.path.basename(job['csv_path'])}"  # Updated path
        
    return jsonify(response)

@app.route('/video_stream/<job_id>')
def video_stream(job_id):
    """Stream video processing frames for a specific job"""
    def generate():
        while job_id in processing_jobs and processing_jobs[job_id]['status'] == 'processing':
            if job_id in stream_queues:
                try:
                    frame = stream_queues[job_id].get(timeout=0.5)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except queue.Empty:
                    # No new frames available, wait a bit
                    time.sleep(0.1)
            else:
                # Queue not set up yet, wait
                time.sleep(0.1)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download_csv/<job_id>')
def download_csv(job_id):
    """Download detection results CSV"""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    
    if 'csv_path' not in job:
        return jsonify({'error': 'CSV file not found'}), 404
    
    return send_file(job['csv_path'], as_attachment=True, download_name='detection_results.csv')

# This is needed for Vercel serverless function
app.debug = False

# For local development
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)