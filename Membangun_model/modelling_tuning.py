import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dagshub

# --- KONFIGURASI DAGSHUB ---
REPO_OWNER = "raflikamal" 
REPO_NAME = "SMSML_M-Rafli-Kamal" 

dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

# --- FUNGSI PELATIHAN & LOGGING ---
def train_and_log_model():
    print("Memulai proses training...")
    
    # 1. Load Data
    try:
        df = pd.read_csv("loan_data_cleaned_automated.csv")
    except FileNotFoundError:
        print("Error: File dataset 'loan_data_cleaned_automated.csv' tidak ditemukan di folder ini.")
        return

    # 2. Split Data
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Setup MLflow Experiment
    experiment_name = "Loan_Approval_Classification_Advance"
    try:
        mlflow.set_experiment(experiment_name)
    except:
        print(f"Eksperimen {experiment_name} mungkin sudah ada atau terjadi error saat pembuatan.")

    with mlflow.start_run(run_name="RandomForest_ManualLogging"):
        
        # --- HYPERPARAMETERS ---
        n_estimators = 100
        max_depth = 10
        random_state = 42
        
        # Log Parameters (Manual)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        # --- MODEL TRAINING ---
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        
        # --- PREDIKSI & METRIK ---
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary') 
        rec = recall_score(y_test, y_pred, average='binary')     
        f1 = f1_score(y_test, y_pred, average='binary')         
        
        print(f"Accuracy: {acc}")
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1-Score: {f1}")

        # Log Metrics (Manual - Sesuai Autolog)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # --- ARTEFAK TAMBAHAN (Syarat Advance: Minimal 2) ---
        
        # 1. Confusion Matrix Plot
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()
        
        mlflow.log_artifact("confusion_matrix.png") 
        
        # 2. Feature Importance Plot
        if hasattr(model, 'feature_importances_'):
            feature_importances = pd.Series(model.feature_importances_, index=X.columns)
            plt.figure(figsize=(10, 6))
            feature_importances.nlargest(10).plot(kind='barh')
            plt.title('Top 10 Feature Importances')
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            plt.close()
            
            mlflow.log_artifact("feature_importance.png")
        else:
            print("Model tidak memiliki atribut feature_importances_.")
        
        # Log Model (Manual)
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print("Training selesai. Metrik dan artefak tersimpan di DagsHub MLflow.")
        
        # Bersihkan file gambar lokal
        if os.path.exists("confusion_matrix.png"):
            os.remove("confusion_matrix.png")
        if os.path.exists("feature_importance.png"):
            os.remove("feature_importance.png")

if __name__ == "__main__":
    # Pastikan Anda sudah login DagsHub di terminal atau set env variable
    # dagshub login
    train_and_log_model()