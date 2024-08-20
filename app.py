from flask import Flask, request, jsonify
from fiber_identification import identify_fiber_in_image
from strokes import identify_stroke_in_image
from elevation import extract_features  # Import the extract_features function
from elevation_infusion import color_features_infusion_predict  # Import the extract_features function
from PIL import Image
import io
import base64
import cv2
import numpy as np
import joblib
import os
from predictor import predict_tea_variant

app = Flask(__name__)

# Load the model and label encoder
model_path = 'models/'
rf_color = joblib.load(os.path.join(model_path, 'rf_color.pkl'))
color_label_encoder = joblib.load(os.path.join(model_path, 'color_label_encoder.pkl'))
# Load the infusion model and label encoder
infusion_rf = joblib.load(os.path.join(model_path, 'rf_color_infusion.pkl'))
infusion_label_encoder = joblib.load(os.path.join(model_path, 'color_label_encoder_infusion.pkl'))
# Load the RandomForest model and label encoder
model_path = 'models/'
size_rf = joblib.load(os.path.join(model_path, 'rf_size.pkl'))
size_label_encoder = joblib.load(os.path.join(model_path, 'size_label_encoder.pkl'))


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Read image
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Extract features
    features = extract_features(image)

    # Predict
    y_pred_rf = rf_color.predict([features])
    t_quality = color_label_encoder.inverse_transform([y_pred_rf[0]])

    return jsonify({"tea_elevation": t_quality[0]})

@app.route('/predict-infusion', methods=['POST'])
def predict_infusion():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Read image
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Extract features
    features = color_features_infusion_predict(image)

    # Predict
    y_pred_rf = infusion_rf.predict([features])
    t_quality = infusion_label_encoder.inverse_transform([y_pred_rf[0]])

    return jsonify({"tea_elevation": t_quality[0]})



@app.route('/identify-fiber', methods=['POST'])
def identify_fiber():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Convert image to OpenCV format
    image = Image.open(file).convert('RGB')
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Process image for fiber identification
    result_image, stats = identify_fiber_in_image(image)
    
    # Convert result image to PIL format for response
    result_image_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    result_image_pil.save(buffered, format="JPEG")
    result_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        "statistics": stats,
        "result_image": result_image_base64
    })

@app.route('/identify-stroke', methods=['POST'])
def identify_stroke():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Convert image to OpenCV format
    image = Image.open(file).convert('RGB')
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Process image for stroke identification
    result_image, stats = identify_stroke_in_image(image)
    
    # Convert result image to PIL format for response
    result_image_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    result_image_pil.save(buffered, format="JPEG")
    result_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        "statistics": stats,
        "result_image": result_image_base64
    })


@app.route('/predict-tea-variant', methods=['POST'])
def predict_tea_variant_route():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Convert image to PIL format
    image = Image.open(file).convert('RGB')

    # Predict tea variant
    t_variant = predict_tea_variant(image)

    return jsonify({"tea_variant": t_variant})

if __name__ == '__main__':
    app.run(debug=True, port=8080)