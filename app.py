from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Charger les modèles au démarrage
MODEL_PATH = 'diabetes_model.pkl'
SCALER_PATH = 'diabetes_scaler.pkl'
FEATURES_PATH = 'feature_names.pkl'

# Vérifier que les fichiers existent
if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
    raise FileNotFoundError("Les fichiers du modèle sont manquants!")

# Charger le modèle, scaler et feature names
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

with open(FEATURES_PATH, 'rb') as f:
    feature_names = pickle.load(f)

print("✅ Modèle chargé avec succès!")
print(f"📋 Features: {feature_names}")


@app.route('/')
def home():
    """Page d'accueil avec le formulaire"""
    return render_template('index.html', features=feature_names)


@app.route('/predict', methods=['POST'])
def predict():
    """Route pour la prédiction"""
    try:
        # Récupérer les données du formulaire
        data = request.get_json()
        
        # Extraire les valeurs dans l'ordre des features
        values = []
        for feature in feature_names:
            value = float(data.get(feature, 0))
            values.append(value)
        
        # Créer l'array numpy
        patient_data = np.array([values])
        
        # Standardiser les données
        patient_scaled = scaler.transform(patient_data)
        
        # Faire la prédiction
        prediction = model.predict(patient_scaled)[0]
        
        # Obtenir les probabilités si disponible
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(patient_scaled)[0]
            prob_non_diabetic = float(probabilities[0])
            prob_diabetic = float(probabilities[1])
        else:
            prob_non_diabetic = None
            prob_diabetic = None
        
        # Préparer la réponse
        response = {
            'success': True,
            'prediction': int(prediction),
            'prediction_text': 'DIABÉTIQUE' if prediction == 1 else 'NON DIABÉTIQUE',
            'probability_non_diabetic': prob_non_diabetic,
            'probability_diabetic': prob_diabetic,
            'patient_data': {feature: values[i] for i, feature in enumerate(feature_names)}
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/health')
def health():
    """Route de vérification de santé de l'application"""
    return jsonify({
        'status': 'OK',
        'model_loaded': model is not None,
        'features': feature_names
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)