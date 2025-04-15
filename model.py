import joblib
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Sample data for demonstration
symptoms = ['fever', 'headache', 'cough', 'fatigue', 'sore throat', 'runny nose', 'shortness of breath', 'chest pain', 'nausea', 'vomiting']
diseases = ['Cold', 'Flu', 'Migraine', 'Asthma']

# Generate synthetic training data
X = np.random.randint(2, size=(100, len(symptoms)))
y = np.random.choice(diseases, size=100)

clf = DecisionTreeClassifier()
clf.fit(X, y)

# Save model and symptom list
joblib.dump(clf, "model.pkl")
joblib.dump(symptoms, "symptom_list.pkl")

print("Model and symptom list saved successfully.")
