"""Configuration du projet MedTriage AI"""
from pathlib import Path

# Chemins
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = FIGURES_DIR / "metrics"

# Créer les répertoires s'ils n'existent pas
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset
DATASET_PATH = DATA_RAW / "Disease_symptom_and_patient_profile_dataset.csv"

# Colonnes
SYMPTOM_COLUMNS = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
PROFILE_COLUMNS = ['Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']
FEATURE_COLUMNS = SYMPTOM_COLUMNS + PROFILE_COLUMNS
TARGET_COLUMN = 'Outcome Variable'
DISEASE_COLUMN = 'Disease'

# Catégories à encoder
CATEGORICAL_COLUMNS = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 
                       'Gender', 'Blood Pressure', 'Cholesterol Level']

# Parametres du modèle
RANDOM_STATE = 42
TEST_SIZE = 0.2
TRAIN_SIZE = 0.8

# Hyperparametres
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
}

# Fichiers de sortie
SCALER_PATH = DATA_PROCESSED / "scaler.pkl"
ENCODER_PATH = DATA_PROCESSED / "encoder.pkl"
MODEL_PATH = RESULTS_DIR / "model.pkl"
