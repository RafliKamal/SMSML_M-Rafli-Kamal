import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_basic_model():
    print("Memulai proses training (Basic - Autolog)...")
    
    # 1. Load Data
    try:
        df = pd.read_csv("loan_data_cleaned_automated.csv")
    except FileNotFoundError:
        print("Error: File dataset 'loan_data_cleaned_automated.csv' tidak ditemukan.")
        return

    # 2. Split Data
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Setup MLflow Experiment (Lokal)
    mlflow.set_tracking_uri("http://127.0.0.1:5000") 
    mlflow.set_experiment("Loan_Approval_Classification_Basic")

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_Autolog"):
        
        # --- MODEL TRAINING ---
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # --- PREDIKSI ---
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Training selesai. Accuracy: {acc}")
        print("Semua log (param, metric, model) sudah dicatat otomatis oleh Autolog.")

if __name__ == "__main__":
    train_basic_model()