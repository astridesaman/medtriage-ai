"""Entraînement du modèle de classification"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from pathlib import Path
from . import config
from .features import FeatureEngineer

class ModelTrainer:
    """Classe pour l'entraînement du modèle"""
    
    def __init__(self):
        self.model = RandomForestClassifier(**config.MODEL_PARAMS)
        self.feature_engineer = FeatureEngineer()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self, df):
        """Prépare les données pour l'entraînement"""
        # Prétraiter les features
        df_processed = self.feature_engineer.fit_transform(df)
        
        # Séparer features et cible
        X = df_processed[config.FEATURE_COLUMNS + ['respiratory_issues', 'age_group']]
        y = df_processed[config.TARGET_COLUMN]
        
        # Encoder la variable cible
        target_encoder = pd.factorize(y)[1]
        y_encoded = pd.factorize(y)[0]
        
        # Diviser en train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE,
            stratify=y_encoded
        )
        
        # Scaler les features
        self.X_train, self.X_test = self.feature_engineer.scale_features(
            self.X_train, self.X_test
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train(self):
        """Entraîne le modèle"""
        print("🚀 Entraînement du modèle...")
        self.model.fit(self.X_train, self.y_train)
        
        # Évaluation sur l'ensemble d'entraînement
        train_accuracy = self.model.score(self.X_train, self.y_train)
        print(f"✓ Accuracy (train): {train_accuracy:.4f}")
        
        # Évaluation sur l'ensemble de test
        test_accuracy = self.model.score(self.X_test, self.y_test)
        print(f"✓ Accuracy (test): {test_accuracy:.4f}")
        
        # Validation croisée
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        print(f"✓ Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self.model
    
    def predict(self, X):
        """Fait des prédictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Prédictions avec probabilités"""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Récupère l'importance des features"""
        feature_names = list(self.X_train.columns)
        importances = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, path=None):
        """Sauvegarde le modèle"""
        if path is None:
            path = config.MODEL_PATH
        joblib.dump(self.model, path)
        print(f"✓ Modèle sauvegardé: {path}")
    
    def load_model(self, path=None):
        """Charge un modèle sauvegardé"""
        if path is None:
            path = config.MODEL_PATH
        self.model = joblib.load(path)
        print(f"✓ Modèle chargé: {path}")