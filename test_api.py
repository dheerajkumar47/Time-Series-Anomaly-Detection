import requests
import time
import sys

API_URL = "http://localhost:8000"

def test_health():
    print("Testing /health...")
    try:
        response = requests.get(f"{API_URL}/health")
        assert response.status_code == 200
        print("✅ Health check passed")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        sys.exit(1)

def test_data_generation():
    print("Testing /data/generate...")
    payload = {
        "data_type": "Financial Transactions",
        "n_points": 100,
        "anomaly_rate": 0.1
    }
    response = requests.post(f"{API_URL}/data/generate", json=payload)
    if response.status_code != 200:
        print(f"❌ Data generation failed: {response.text}")
        sys.exit(1)
    data = response.json()
    assert data["total_points"] > 0
    print("✅ Data generation passed")

def test_training():
    print("Testing /models/train...")
    payload = {
        "use_prophet": True,
        "use_isolation_forest": True,
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": True,
        "seasonality_mode": "additive",
        "contamination": 0.05,
        "window_size": 24
    }
    response = requests.post(f"{API_URL}/models/train", json=payload)
    if response.status_code != 200:
        print(f"❌ Training failed: {response.text}")
        sys.exit(1)
    result = response.json()
    assert "Prophet" in result["models"]
    assert "Isolation Forest" in result["models"]
    print("✅ Training passed")

def test_results():
    print("Testing /models/results...")
    response = requests.get(f"{API_URL}/models/results")
    if response.status_code != 200:
        print(f"❌ Get results failed: {response.text}")
        sys.exit(1)
    results = response.json()
    assert "Prophet" in results
    assert "Isolation Forest" in results
    print("✅ Results retrieval passed")

if __name__ == "__main__":
    # Wait for server to start
    print("Waiting for server to start...")
    for i in range(10):
        try:
            requests.get(f"{API_URL}/health")
            break
        except:
            time.sleep(1)
    else:
        print("❌ Server failed to start")
        sys.exit(1)

    test_health()
    test_data_generation()
    test_training()
    test_results()
    print("🎉 All tests passed!")
