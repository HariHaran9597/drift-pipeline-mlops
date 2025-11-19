# scripts/generate_traffic.py
import requests
import time
import random
import sys

URL = "http://localhost:8000/predict"

def send_traffic():
    print(f"Blasting traffic to {URL}...")
    print("Press CTRL+C to stop.")
    
    count = 0
    while True:
        try:
            # Random weather data
            payload = {
                "temperature": random.uniform(20, 45),
                "humidity": random.uniform(30, 90)
            }
            
            response = requests.post(URL, json=payload)
            
            if response.status_code == 200:
                pred = response.json()['predicted_demand']
                sys.stdout.write(f"\rRequest {count}: Demand {pred:.2f} | Status: 200")
                sys.stdout.flush()
            else:
                print(f"\nError: {response.status_code}")
                
            count += 1
            time.sleep(0.5) # Send 2 requests per second
            
        except Exception as e:
            print(f"\nConnection Error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    send_traffic()