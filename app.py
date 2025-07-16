from flask import Flask, render_template, request, jsonify
import joblib
import json
import random

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('mental_health_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load intent-response data
with open('data.json') as f:
    data = json.load(f)

# Create a dictionary: tag -> responses
response_dict = {intent['tag']: intent['responses'] for intent in data['intents']}

# ✅ ✅ ✅ ROOT ROUTE — MUST BE PRESENT
@app.route('/')
def index():
    return render_template('chat.html')  # Make sure templates/chat.html exists

# ✅ Chatbot Response Route
@app.route('/get_response', methods=['POST'])
def get_chat_response():
    try:
        user_input = request.json['message']
        input_vec = vectorizer.transform([user_input])
        intent = model.predict(input_vec)[0]
        responses = response_dict.get(intent, ["I'm here to listen."])
        return jsonify({'response': random.choice(responses)})
    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({'response': "Oops. Something went wrong on the server."})

if __name__ == '__main__':
    app.run(debug=True)
