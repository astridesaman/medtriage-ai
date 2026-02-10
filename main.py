"""Pipeline principal - Orchestration du projet MedTriage AI"""
import pandas as pd
from pathlib import Path
from src.config import DATASET_PATH
from src.trainer import ModelTrainer
from src.evaluate import ModelEvaluator
from src.explain import ModelExplainer

def main():
    """Fonction principale du pipeline"""
    
    print("\n" + "="*60)
    print("🏥 MEDTRIAGE AI - Pipeline Complet")
    print("="*60)
    

    # 1. CHARGEMENT DES DONNÉES
    
    print("\n📂 Étape 1: Chargement des données...")
    df = pd.read_csv(DATASET_PATH)
    print(f"✓ Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # 2. PRÉPARATION DES DONNÉES
    
    print("\n🔧 Étape 2: Préparation des données...")
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    print(f"✓ Train set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    
    
    # 3. ENTRAÎNEMENT DU MODÈLE
    
    print("\n🚀 Étape 3: Entraînement du modèle...")
    trainer.train()
    
    
    # 4. ÉVALUATION DU MODÈLE
    
    print("\n📊 Étape 4: Évaluation du modèle...")
    evaluator = ModelEvaluator(trainer.model, X_test, y_test)
    metrics = evaluator.evaluate()
    
    # Générer les visualisations
    print("\n📈 Génération des visualisations...")
    evaluator.plot_confusion_matrix()
    evaluator.plot_feature_importance()
    evaluator.plot_roc_curve()
    evaluator.save_metrics_report()
    
    
    # 5. EXPLICATION DU MODÈLE
    
    print("\n🔍 Étape 5: Explication du modèle...")
    explainer = ModelExplainer(trainer.model, X_train, list(X_train.columns))
    
    # Afficher l'importance des features
    explainer.feature_importance_summary()
    explainer.plot_feature_importance_detailed()
    explainer.get_decision_path_stats()
    
    
    # 6. SAUVEGARDE DU MODÈLE
    
    print("\n💾 Étape 6: Sauvegarde du modèle...")
    trainer.save_model()
    
    
    # RÉSUMÉ FINAL
    
    print("\n" + "="*60)
    print("✅ PIPELINE TERMINÉ AVEC SUCCÈS!")
    print("="*60)
    print(f"\n📊 Résultats Finaux:")
    print(f"   • Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   • Precision: {metrics['precision']:.4f}")
    print(f"   • Recall:    {metrics['recall']:.4f}")
    print(f"   • F1-Score:  {metrics['f1']:.4f}")
    print(f"\n📂 Fichiers générés:")
    print(f"   • Modèle: results/model.pkl")
    print(f"   • Métriques: results/figures/metrics/")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
