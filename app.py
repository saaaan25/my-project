from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# 1. Crear carpetas primero para que el calentamiento pueda usarlas
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 2. Cargar el modelo
interpreter = tf.lite.Interpreter(model_path='brain_tumor_cnn.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- TRUCO DE CALENTAMIENTO TOTAL ---
# Obligamos a TensorFlow y a Matplotlib a procesar todo en el arranque.
try:
    print("Iniciando calentamiento de TensorFlow y Matplotlib...")
    # Simular una predicción
    forma_entrada = input_details[0]['shape']
    imagen_falsa = np.zeros(forma_entrada, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], imagen_falsa)
    interpreter.invoke()
    
    # Simular la creación del gráfico (esto es lo que causaba el timeout)
    plt.figure(figsize=(6, 4))
    plt.bar(['test'], [1.0], color='skyblue')
    plt.title('Calentamiento')
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'warmup.png'))
    plt.close()
    print("Calentamiento completado. El servidor está listo para la acción.")
except Exception as e:
    print(f"Error en calentamiento: {e}")
# ----------------------------------------

def predict_with_tflite(img_array):
    img_array = img_array.astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

@app.route('/api/clasificar', methods=['POST'])
def clasificar_api():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No se envió imagen'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = predict_with_tflite(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    probabilities = {
        class_names[i]: float(f"{prob:.4f}")
        for i, prob in enumerate(prediction)
    }

    plt.figure(figsize=(6, 4))
    plt.bar(probabilities.keys(), probabilities.values(), color='skyblue')
    plt.title('Probabilidades por clase')
    plt.ylabel('Confianza')
    plt.tight_layout()
    graph_path = os.path.join(app.config['UPLOAD_FOLDER'], 'probabilidades.png')
    plt.savefig(graph_path)
    plt.close()

    return jsonify({
        'prediction': f'Predicción: {predicted_class.upper()}',
        'image_name': filename,
        'probs': probabilities
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)