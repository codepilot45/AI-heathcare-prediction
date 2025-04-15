from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and symptoms list
model = joblib.load('model.pkl')
symptoms = joblib.load('symptom_list.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['symptoms']
    
    # Process the input (turn it into a list of symptoms)
    user_symptoms = user_input.lower().split(',')
    user_symptoms = [symptom.strip() for symptom in user_symptoms]
    
    # Create a feature vector based on the user's symptoms
    feature_vector = [1 if symptom in user_symptoms else 0 for symptom in symptoms]
    
    # Predict using the model
    prediction = model.predict([feature_vector])
    
    # Render the result page with prediction
    return render_template('result.html', symptoms=user_input, prediction=prediction[0])

# Ensure the app only runs when executed directly
if __name__ == '__main__':
    app.run(debug=True)
