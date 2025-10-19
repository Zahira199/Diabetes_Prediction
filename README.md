# Diabetes Prediction Project

## Description

Ce projet permet de prédire si une personne est diabétique ou non à partir de données cliniques. Le modèle utilisé est un **Random Forest** entraîné sur le **Pima Indians Diabetes Dataset**. L'application est déployée avec **Flask**, avec une interface web simple pour saisir les données et obtenir une prédiction.

## Contenu du projet

| Fichier / Dossier | Description |
|-------------------|-------------|
| `.venv/` | Environnement virtuel Python (ignoré sur GitHub) |
| `app.py` | Script principal Flask pour l'application web |
| `diabetes_model.pkl` | Modèle Random Forest sauvegardé |
| `diabetes_scaler.pkl` | Scaler pour standardiser les données |
| `feature_names.pkl` | Liste des noms des features utilisées par le modèle |
| `Diabetes_Prediction.ipynb` | Notebook contenant l'exploration, l'entraînement et l'évaluation du modèle |
| `templates/` | Dossier contenant le fichier HTML pour l'interface web |

## Prérequis

- Python 3.8 ou supérieur  
- pip (gestionnaire de packages Python)

Installer les packages nécessaires :

```bash
pip install pandas numpy scikit-learn matplotlib seaborn flask
```

## Utilisation

1. **Activer l'environnement virtuel** (si nécessaire) :

```bash
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

2. **Lancer l'application Flask** :

```bash
python app.py
```

3. **Ouvrir le navigateur** :

```
http://127.0.0.1:5000/
```

4. **Saisir les valeurs cliniques** dans le formulaire pour obtenir la prédiction (`Diabétique` ou `Non Diabétique`).

## Performances du modèle

**Modèle sélectionné** : Random Forest Classifier

| Métrique | Valeur |
|----------|--------|
| Accuracy | 77.9% |
| F1-Score | 0.66 |
| AUC-ROC | 0.82 |

**Variables les plus importantes** :
1. Glucose
2. BMI
3. DiabetesPedigreeFunction
4. Age

## Structure du projet

```
Diabet_predection/
├── .venv/
├── app.py
├── diabetes_model.pkl
├── diabetes_scaler.pkl
├── feature_names.pkl
├── Diabetes_Prediction.ipynb
├── templates/
│   └── index.html
└── README.md
```

## Auteur

**Zahira Elaamrani**  
MSDAI 2ème Année

## Licence

MIT License