import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Load the YOLOv8 model from the current directory
try:
    model = YOLO('best.pt')
    print("YOLOv8 model 'best.pt' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the model was loaded successfully
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 503
    
    # Check if an image file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400
    
    file = request.files['image']
    
    # Check for valid file extensions
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file type. Please upload a PNG or JPG image.'}), 400
    
    try:
        # Open the image file
        img = Image.open(file.stream)
        
        # Run inference with the model
        results = model(img)

        # Process the results
        predictions = []
        # 'names' is a dictionary containing class names, like the ones in your data.pkl file.
        class_names = results[0].names 

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                label = class_names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                predictions.append({
                    'label': label,
                    'confidence': confidence,
                    'box': bbox
                })
        
        return jsonify({'predictions': predictions}), 200

    except Exception as e:
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == "__main__":
    print("Attempting to run Flask app...") # Added this line
    app.run(host="0.0.0.0", port=5000, debug=True)