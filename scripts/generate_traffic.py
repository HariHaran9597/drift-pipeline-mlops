# scripts/generate_traffic.py
#!/usr/bin/env python3
# scripts/generate_traffic.py
import requests
import time
import random
import sys
from datetime import datetime

URL = "http://localhost:8000/predict"
TEMP_MIN, TEMP_MAX = 20, 45
HUMIDITY_MIN, HUMIDITY_MAX = 30, 90
REQUEST_INTERVAL = 0.5  # 2 requests per second

def send_traffic():
    print(f"{'='*60}")
    print(f"Traffic Generator Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: {URL}")
    print(f"Rate: {1/REQUEST_INTERVAL:.1f} requests/sec")
    print(f"Temperature Range: {TEMP_MIN}-{TEMP_MAX}°C")
    print(f"Humidity Range: {HUMIDITY_MIN}-{HUMIDITY_MAX}%")
    print(f"{'='*60}")
    print("Press CTRL+C to stop.\n")
    
    count = 0
    errors = 0
    start_time = time.time()
    
    while True:
        try:
            # Random weather data
            payload = {
                "temperature": random.uniform(TEMP_MIN, TEMP_MAX),
                "humidity": random.uniform(HUMIDITY_MIN, HUMIDITY_MAX)
            }
            
            response = requests.post(URL, json=payload, timeout=5)
            
            if response.status_code == 200:
                pred = response.json()['predicted_demand']
                elapsed = time.time() - start_time
                success_rate = ((count - errors) / max(1, count)) * 100
                sys.stdout.write(
                    f"\r[{elapsed:7.1f}s] Req #{count}: Demand={pred:7.2f} | "
                    f"Success={success_rate:.1f}% ({count-errors}/{count})"
                )
                sys.stdout.flush()
            else:
                errors += 1
                print(f"\n✗ Error {response.status_code}: {response.text[:100]}")
                
            count += 1
            time.sleep(REQUEST_INTERVAL)
            
        except requests.exceptions.Timeout:
            errors += 1
            print(f"\n✗ Timeout: Request took >5 seconds")
            time.sleep(2)
        except requests.exceptions.ConnectionError as ce:
            errors += 1
            print(f"\n✗ Connection Error: {str(ce)[:100]}")
            print("  → Is the API running? Try: docker-compose up -d")
            time.sleep(3)
        except KeyboardInterrupt:
            print(f"\n\n{'='*60}")
            print(f"Traffic Generator Stopped at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            elapsed = time.time() - start_time
            print(f"Total Requests: {count}")
            print(f"Successful: {count - errors}")
            print(f"Failed: {errors}")
            print(f"Success Rate: {((count-errors)/max(1,count))*100:.1f}%")
            print(f"Duration: {elapsed:.1f}s")
            print(f"Avg Rate: {count/elapsed:.2f} req/s")
            print(f"{'='*60}")
            break
        except Exception as e:
            errors += 1
            print(f"\n✗ Unexpected error: {str(e)[:100]}")
            time.sleep(2)

if __name__ == "__main__":
    send_traffic()