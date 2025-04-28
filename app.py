from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load models
detection_model = tf.keras.models.load_model('skin_disease_model.keras')

with open('chatbot_data.pkl', 'rb') as f:
    chatbot_data = pickle.load(f)

qa_pairs = chatbot_data['qa_pairs']
embeddings = chatbot_data['embeddings']

# Load sentence transformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Disease classes (you must match according to your dataset)
class_names = ['Disease_1', 'Disease_2', 'Disease_3', 'Disease_4', 'Disease_5', 'Disease_6']

# Routes

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded!", 400

    file = request.files['image']
    img = tf.keras.preprocessing.image.load_img(file, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = detection_model.predict(img_array)
    predicted_class_idx = np.argmax(prediction)
    predicted_class = class_names[predicted_class_idx]

    return render_template('upload.html', prediction=predicted_class)

# Chatbot route
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json['question']

    user_embedding = embedder.encode([user_input])
    similarities = cosine_similarity(user_embedding, embeddings)[0]
    best_idx = np.argmax(similarities)
    answer = qa_pairs[best_idx]['answer']

    return jsonify({'answer': answer})

# Main
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
