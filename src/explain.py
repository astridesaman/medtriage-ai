"""Explication et interprétabilité du modèle"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import config

class ModelExplainer:
    """Classe pour l'explication du modèle"""
    
    def __init__(self, model, X_train, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or list(X_train.columns)
        
    def get_feature_importance(self):
        """Retourne l'importance des features"""
        importances = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def explain_prediction(self, X_instance, prediction, probability=None):
        """Explique une prédiction spécifique"""
        print("\n" + "="*50)
        print("🔍 EXPLICATION DE LA PRÉDICTION")
        print("="*50)
        print(f"Prédiction: Classe {prediction}")
        if probability is not None:
            print(f"Confiance: {max(probability)*100:.2f}%")
        
        print("\nValeurs des features:")
        for i, feature in enumerate(self.feature_names):
            print(f"  {feature}: {X_instance[i]:.4f}")
    
    def plot_feature_importance_detailed(self, top_n=15, save_path=None):
        """Trace un graphique détaillé de l'importance des features"""
        importance_df = self.get_feature_importance().head(top_n)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(importance_df)), importance_df['importance'].values)
        
        # Colorer les barres
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.yticks(range(len(importance_df)), importance_df['feature'].values)
        plt.xlabel('Importance')
        plt.title(f'Feature Importance (Top {top_n})')
        plt.tight_layout()
        
        if save_path is None:
            save_path = config.METRICS_DIR / "feature_importance_detailed.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance détaillée sauvegardée: {save_path}")
        plt.close()
    
    def feature_importance_summary(self):
        """Affiche un résumé de l'importance des features"""
        importance_df = self.get_feature_importance()
        
        print("\n" + "="*50)
        print("📊 IMPORTANCE DES FEATURES (Top 10)")
        print("="*50)
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df
    
    def get_decision_path_stats(self):
        """Obtient des statistiques sur les chemins de décision"""
        # Cette fonction montre des informations générales sur le modèle
        print("\n" + "="*50)
        print("🌳 STATISTIQUES DU MODÈLE")
        print("="*50)
        print(f"Type: {type(self.model).__name__}")
        print(f"Nombre d'arbres: {self.model.n_estimators}")
        print(f"Profondeur max: {self.model.max_depth}")
        print(f"Features utilisées: {len(self.feature_names)}")