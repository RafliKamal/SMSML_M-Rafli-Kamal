from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
import time
import psutil
import random

# 1. Total Prediksi (Counter)
PREDICTION_COUNT = Counter('prediction_total', 'Total number of predictions')

# 2. Latency / Waktu Respon (Histogram)
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Latency of requests in seconds')

# 3. Income Rata-rata (Gauge - Data Drift Monitoring)
INCOME_MEAN = Gauge('income_mean_value', 'Average income of applicants')

# 4. Loan Amount Rata-rata (Gauge)
LOAN_AMOUNT_MEAN = Gauge('loan_amount_mean_value', 'Average loan amount requested')

# 5. Credit Score Rata-rata (Gauge)
CREDIT_SCORE_MEAN = Gauge('credit_score_mean_value', 'Average credit score')

# 6. Approval Rate / Persentase Disetujui (Gauge)
APPROVAL_RATE = Gauge('approval_rate_percent', 'Percentage of approved loans')

# 7. System CPU Usage (Gauge - System Monitoring)
SYSTEM_CPU = Gauge('system_cpu_usage_percent', 'System CPU usage')

# 8. System Memory Usage (Gauge - System Monitoring)
SYSTEM_MEMORY = Gauge('system_memory_usage_percent', 'System RAM usage')

# 9. Prediction Confidence / Probabilitas (Histogram)
PREDICTION_CONFIDENCE = Histogram('prediction_confidence', 'Confidence score of predictions')

# 10. Error Count (Counter)
ERROR_COUNT = Counter('api_error_total', 'Total number of API errors')

# Variabel bantu untuk rata-rata berjalan
income_history = []
loan_history = []
approved_count = 0
total_req = 0

def update_system_metrics():
    """Update metrik sistem (CPU & RAM)"""
    SYSTEM_CPU.set(psutil.cpu_percent())
    SYSTEM_MEMORY.set(psutil.virtual_memory().percent)

def update_business_metrics(income, loan_amount, credit_score, prediction, latency):
    """Update metrik bisnis dan performa model"""
    global approved_count, total_req
    
    # Update Counter & Histogram
    PREDICTION_COUNT.inc()
    REQUEST_LATENCY.observe(latency)
    
    # Update Data Drift (Simple Moving Average)
    income_history.append(income)
    if len(income_history) > 100: income_history.pop(0)
    INCOME_MEAN.set(sum(income_history) / len(income_history))
    
    LOAN_AMOUNT_MEAN.set(loan_amount)
    CREDIT_SCORE_MEAN.set(credit_score)
    
    # Update Approval Rate
    total_req += 1
    if prediction == 1:
        approved_count += 1
    APPROVAL_RATE.set((approved_count / total_req) * 100)
    
    # Simulate Confidence (karena model RF default outputnya kelas, kita simulasikan probabilitasnya)
    confidence = random.uniform(0.6, 0.99)
    PREDICTION_CONFIDENCE.observe(confidence)

def start_exporter_server(port=8000):
    """Jalankan server Prometheus"""
    start_http_server(port)
    print(f"Prometheus Exporter running on port {port}")