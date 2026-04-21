# file: app.py (Final version with corrected postprocess)

import os
import uuid
import numpy as np
import torch
import cv2
from flask import Flask, request, send_file, jsonify
import traceback

# Import your model definition
from model import SketchKeras

# --- Global Setup & Model Loading ---
app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

FIXED_OUTPUT_PATH = "/home/sty/pyfile/sketchKeras_pytorch/src/output.jpg"
UPLOAD_FOLDER = 'uploads_sketch'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Processing Logic ---
def preprocess(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    highpass = img.astype(np.float32) - blurred.astype(np.float32)
    highpass /= 128.0
    max_val = np.max(np.abs(highpass))
    if max_val > 0:
        highpass /= max_val
    ret = np.zeros((512, 512), dtype=np.float32)
    ret[0:h, 0:w] = highpass
    return np.expand_dims(ret, axis=2)

# --- 关键修改: 修正 postprocess 函数 ---
def postprocess(pred, thresh=0.18, smooth=False):
    assert thresh <= 1.0 and thresh >= 0.0

    # `pred` is already a 2D array [height, width] after squeeze()
    # The np.amax(pred, axis=0) was incorrect for single-channel output and is now removed.
    # pred = np.amax(pred, axis=0) # <--- REMOVED THIS LINE

    # The rest of the logic now correctly operates on the 2D image array
    pred[pred < thresh] = 0
    pred = 1 - pred
    pred *= 255
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    if smooth:
        pred = cv2.medianBlur(pred, 3)
    return pred

# --- Load the Model ---
try:
    print("Loading SketchKeras model...")
    WEIGHT_PATH = "/home/sty/pyfile/sketchKeras_pytorch/weights/model.pth" 
    model = SketchKeras().to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.eval()
    print(f"✅ SketchKeras model loaded successfully from: {WEIGHT_PATH}")
except Exception as e:
    print(f"❌ Error loading SketchKeras model: {e}")
    model = None

# --- API Route Definition ---
@app.route('/sketchify', methods=['POST'])
def sketchify_image():
    if model is None:
        return jsonify({"error": "Model is not loaded on the server."}), 500
    if 'image' not in request.files:
        return jsonify({"error": "No 'image' file part in the request."}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected for uploading."}), 400

    temp_input_filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
    input_path = os.path.join(UPLOAD_FOLDER, temp_input_filename)
    file.save(input_path)
    
    try:
        img = cv2.imread(input_path)
        if img is None:
            return jsonify({"error": "Could not read the uploaded image file."}), 400

        height, width = float(img.shape[0]), float(img.shape[1])
        if width > height:
            new_width, new_height = (512, int(512 / width * height))
        else:
            new_width, new_height = (int(512 / height * width), 512)
        resized_img = cv2.resize(img, (new_width, new_height))
        
        processed_img = preprocess(resized_img)
        
        x = processed_img.reshape(1, *processed_img.shape).transpose(0, 3, 1, 2)
        x = torch.from_numpy(x).float().to(device)
        
        with torch.no_grad():
            pred_tensor = model(x)
        
        # Squeeze the tensor. If shape is [1, 1, 512, 512], it becomes [512, 512]
        squeezed_pred = pred_tensor.squeeze()
        
        # Now, the input to postprocess is a 2D numpy array
        output_2d = postprocess(squeezed_pred.cpu().detach().numpy(), thresh=0.18, smooth=False) 
        
        # Crop the 2D array
        output = output_2d[:new_height, :new_width]

        if output.size == 0:
            return jsonify({"error": "Model processing resulted in an empty image."}), 500

        success = cv2.imwrite(FIXED_OUTPUT_PATH, output)
        
        if not success:
            return jsonify({"error": "Failed to save the processed image on the server."}), 500
        
        print(f"Image successfully overwritten at: {FIXED_OUTPUT_PATH}")

        return send_file(FIXED_OUTPUT_PATH, mimetype='image/jpeg')

    except Exception as e:
        print(f"An error occurred during processing:")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500
    
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

# Health check route
@app.route('/', methods=['GET'])
def index():
    return "SketchKeras API (Final Corrected) is running!"

# Start the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)