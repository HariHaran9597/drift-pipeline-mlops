# test_api.py
import requests
import json

url = "http://localhost:8000/predict"
payload = {
    "temperature": 35.5, 
    "humidity": 40.0
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("Success!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Failed with status {response.status_code}")
        print(f"Error details: {response.text}")
        
except Exception as e:
    print(f"Connection failed: {e}")
    print("Make sure docker container is running!")