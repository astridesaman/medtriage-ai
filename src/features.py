"""Ingénierie des features et prétraitement des données"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
from . import config

class FeatureEngineer:
    """Classe pour le prétraitement et l'ingénierie des features"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def encode_categorical(self, df, columns=None):
        """Encode les variables catégoriques en numériques"""
        if columns is None:
            columns = config.CATEGORICAL_COLUMNS
            
        df_encoded = df.copy()
        
        for col in columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df_encoded[col] = self.label_encoders[col].transform(df[col])
                
        return df_encoded
    
    def scale_features(self, X_train, X_test=None):
        """Normalise les features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def create_features(self, df):
        """Crée des features supplémentaires"""
        df_features = df.copy()
        
        # Feature: combinaison de symptômes respiratoires
        df_features['respiratory_issues'] = (
            ((df_features['Cough'] == 'Yes') | 
             (df_features['Difficulty Breathing'] == 'Yes')).astype(int)
        )
        
        # Feature: catégorie d'âge
        df_features['age_group'] = pd.cut(df_features['Age'], 
                                          bins=[0, 18, 35, 50, 65, 100],
                                          labels=['child', 'young_adult', 'adult', 'senior', 'elderly'])
        
        # Encoder la catégorie d'âge
        age_encoder = LabelEncoder()
        df_features['age_group'] = age_encoder.fit_transform(df_features['age_group'])
        
        return df_features
    
    def fit_transform(self, df):
        """Applique tout le prétraitement en une seule étape"""
        # Créer les features
        df = self.create_features(df)
        
        # Encoder les variables catégoriques
        df = self.encode_categorical(df)
        
        return df
    
    def transform(self, df):
        """Applique le prétraitement sans réentrainer"""
        # Créer les features
        df = self.create_features(df)
        
        # Encoder les variables catégoriques
        df = self.encode_categorical(df)
        
        return df
