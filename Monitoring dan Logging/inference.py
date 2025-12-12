import requests
import time
import pandas as pd
import random
import prometheus_exporter as exporter 

# URL Model Serving (Lokal Docker)
MODEL_URL = "http://localhost:8080/invocations"


try:
    df = pd.read_csv("../Membangun_model/loan_data_cleaned_automated.csv")
    print("Dataset loaded successfully.")
except:
    print("CSV not found")
    df = pd.DataFrame(data)

def run_inference_loop():
    # Jalankan server metrics di background
    exporter.start_exporter_server(8000)
    
    print("Starting inference loop... (Press Ctrl+C to stop)")
    
    while True:
        try:
            # Ambil 1 baris data acak
            sample = df.sample(1)
            
            payload = {"dataframe_split": sample.to_dict(orient="split")}
        
            if 'loan_status' in payload['dataframe_split']['columns']:
                idx = payload['dataframe_split']['columns'].index('loan_status')
                del payload['dataframe_split']['columns'][idx]
                del payload['dataframe_split']['data'][0][idx]

            start_time = time.time()
            
            # Kirim Request ke Docker Model
            response = requests.post(MODEL_URL, json=payload, headers={"Content-Type": "application/json"})
            
            latency = time.time() - start_time
            
            if response.status_code == 200:
                prediction = response.json()['predictions'][0]
                print(f"Prediction: {prediction} | Latency: {latency:.4f}s")
                
                # Update Metrics ke Prometheus Exporter
                exporter.update_business_metrics(
                    income=sample['person_income'].values[0],
                    loan_amount=sample['loan_amnt'].values[0],
                    credit_score=sample['credit_score'].values[0],
                    prediction=prediction,
                    latency=latency
                )
            else:
                print(f"Error: {response.text}")
                exporter.ERROR_COUNT.inc()
                
            # Update System Metrics
            exporter.update_system_metrics()
            
            # Delay acak agar grafik terlihat natural
            time.sleep(random.uniform(0.5, 2.0))
            
        except Exception as e:
            print(f"Connection Error: {e}")
            exporter.ERROR_COUNT.inc()
            time.sleep(5)

if __name__ == "__main__":
    run_inference_loop()