from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf

# CONFIG
MODEL_PATH = "model_c2f.tflite"
app = Flask(__name__)
CORS(app) # <-- HABILITA CORS

# LOAD TFLITE MODEL
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ROUTES
@app.route("/")

def home():
    return "Flask + TFLite API running"

@app.route("/predict/<float:celsius>", methods=["GET"])

def predict_get(celsius):
    # PREPARE INPUT (shape: 1x1)
    input_data = np.array([[celsius]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    fahrenheit = float(output[0][0])
    return jsonify({
        "celsius": celsius,
        "fahrenheit": fahrenheit
    })

# MAIN
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

