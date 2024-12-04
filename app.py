import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from collections import deque
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load the trained model
model = load_model("sign_language_model.h5")

# Define class labels
class_labels = {
    1: "Opaque",
    2: "Red",
    3: "Green",
    4: "Yellow",
    5: "Bright",
    6: "Light-blue",
    7: "Colors",
    8: "Pink",
    9: "Women",
    10: "Enemy",
    11: "Son",
    12: "Man",
    13: "Away",
    14: "Drawer",
    15: "Born",
    16: "Learn",
    17: "Call",
    18: "Skimmer",
    19: "Bitter",
    20: "Sweet milk",
    21: "Milk",
    22: "Water",
    23: "Food",
    24: "Argentina",
    25: "Uruguay",
    26: "Country",
    27: "Last name",
    28: "Where",
    29: "Mock",
    30: "Birthday",
    31: "Breakfast",
    32: "Photo",
    33: "Hungry",
    34: "Map"
}



# Parameters
depth = 16
height = 64
width = 64
channels = 3
frames_queue = deque(maxlen=depth)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    print(data)
    frame_data = data.get("frame")
    # Decode the frame from base64
    frame = base64.b64decode(frame_data.split(",")[1])
    np_frame = np.frombuffer(frame, np.uint8)
    img = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

    # Preprocess frame
    resized_frame = cv2.resize(img, (width, height))
    normalized_frame = resized_frame / 255.0
    frames_queue.append(normalized_frame)

    # Once we have enough frames, make a prediction
    if len(frames_queue) == depth:
        input_clip = np.expand_dims(np.array(frames_queue), axis=0)
        predictions = model.predict(input_clip)
        predicted_class = np.argmax(predictions[0])
        predicted_class = int(predicted_class)
        predicted_label = class_labels[predicted_class]
        return jsonify({"text": predicted_label})

    return jsonify({"text": "Processing..."})

if __name__ == "__main__":
    app.run(debug=True)
