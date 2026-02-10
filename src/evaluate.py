"""Évaluation des performances du modèle"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from . import config

class ModelEvaluator:
    """Classe pour l'évaluation du modèle"""
    
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.y_pred_proba = None
        
    def evaluate(self):
        """Évalue le modèle et affiche les métriques"""
        # Prédictions
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Métriques
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        
        print("\n" + "="*50)
        print("📊 PERFORMANCES DU MODÈLE")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Rapport de classification
        print("\n" + "="*50)
        print("📈 RAPPORT DE CLASSIFICATION")
        print("="*50)
        print(classification_report(self.y_test, self.y_pred))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def plot_confusion_matrix(self, save_path=None):
        """Trace la matrice de confusion"""
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Matrice de Confusion')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path is None:
            save_path = config.METRICS_DIR / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Matrice de confusion sauvegardée: {save_path}")
        plt.close()
    
    def plot_feature_importance(self, top_n=15, save_path=None):
        """Trace l'importance des features"""
        feature_names = list(self.X_test.columns)
        importances = self.model.feature_importances_
        
        # Trier par importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.title(f'Top {top_n} Features Importances')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        if save_path is None:
            save_path = config.METRICS_DIR / "feature_importance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance sauvegardée: {save_path}")
        plt.close()
    
    def plot_roc_curve(self, save_path=None):
        """Trace la courbe ROC (pour problèmes multi-class)"""
        if len(np.unique(self.y_test)) == 2:
            fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba[:, 1])
            roc_auc = roc_auc_score(self.y_test, self.y_pred_proba[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.tight_layout()
            
            if save_path is None:
                save_path = config.METRICS_DIR / "roc_curve.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curve sauvegardée: {save_path}")
        plt.close()
    
    def save_metrics_report(self, save_path=None):
        """Sauvegarde un rapport des métriques"""
        metrics = self.evaluate()
        
        if save_path is None:
            save_path = config.METRICS_DIR / "metrics_report.txt"
        
        with open(save_path, 'w') as f:
            f.write("MÉTRIQUES DU MODÈLE\n")
            f.write("="*50 + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric.upper()}: {value:.4f}\n")
        
        print(f"✓ Rapport sauvegardé: {save_path}")
