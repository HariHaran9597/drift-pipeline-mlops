# test_api.py
import requests
import json

url = "http://localhost:8000/predict"
test_cases = [
    {"temperature": 25.0, "humidity": 50.0},  # Normal conditions
    {"temperature": 35.5, "humidity": 40.0},  # Hot, dry
    {"temperature": 20.0, "humidity": 80.0},  # Cool, humid
]

print("=" * 60)
print("Testing Drift-Pipeline API")
print("=" * 60)

for idx, payload in enumerate(test_cases, 1):
    try:
        print(f"\nTest {idx}: T={payload['temperature']}°C, H={payload['humidity']}%")
        response = requests.post(url, json=payload, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Success!")
            print(f"  Model: {result['model_version']}")
            print(f"  Predicted Demand: {result['predicted_demand']} units")
        else:
            print(f"✗ Failed with status {response.status_code}")
            print(f"  Error: {response.text[:200]}")
            
    except requests.exceptions.Timeout:
        print(f"✗ Request timeout (>5 seconds)")
    except requests.exceptions.ConnectionError:
        print(f"✗ Connection failed. Make sure API is running!")
        print(f"  → Try: docker-compose up -d")
        break
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

print("\n" + "=" * 60)