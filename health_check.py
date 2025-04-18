"""
Health check utility for the Continuous Learning Lab.
"""
import os
import sys
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuration
PORT = int(os.getenv("API_PORT", "8000"))
API_URL = f"http://localhost:{PORT}"
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "models/adapter.safetensors")
LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
METRICS_FILE = os.getenv("METRICS_FILE", "logs/metrics.tsv")

def check_api() -> bool:
    """Check if the API is responding."""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        if response.status_code == 200:
            print(f"✅ API is running: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"❌ API returned status code {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ API is not accessible: {e}")
        return False

def check_adapter() -> bool:
    """Check if the adapter file exists."""
    if Path(ADAPTER_PATH).exists():
        size = Path(ADAPTER_PATH).stat().st_size / (1024 * 1024)  # Size in MB
        last_modified = time.ctime(Path(ADAPTER_PATH).stat().st_mtime)
        print(f"✅ Adapter exists: {ADAPTER_PATH} ({size:.2f} MB, updated {last_modified})")
        return True
    else:
        print(f"❌ Adapter not found: {ADAPTER_PATH}")
        return False

def check_logs() -> bool:
    """Check if log files exist and contain recent entries."""
    log_path = Path(LOG_FILE)
    metrics_path = Path(METRICS_FILE)
    
    log_ok = False
    if log_path.exists():
        log_size = log_path.stat().st_size / 1024  # Size in KB
        last_modified = time.ctime(log_path.stat().st_mtime)
        print(f"✅ Log file exists: {LOG_FILE} ({log_size:.2f} KB, updated {last_modified})")
        log_ok = True
    else:
        print(f"❌ Log file not found: {LOG_FILE}")
    
    metrics_ok = False
    if metrics_path.exists():
        metrics_size = metrics_path.stat().st_size / 1024  # Size in KB
        last_modified = time.ctime(metrics_path.stat().st_mtime)
        print(f"✅ Metrics file exists: {METRICS_FILE} ({metrics_size:.2f} KB, updated {last_modified})")
        metrics_ok = True
    else:
        print(f"❌ Metrics file not found: {METRICS_FILE}")
    
    return log_ok and metrics_ok

def test_inference() -> bool:
    """Test model inference with a simple prompt."""
    try:
        test_prompt = "Explain the concept of machine learning in one sentence."
        response = requests.post(
            f"{API_URL}/chat",
            json={"prompt": test_prompt, "topic": "test"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Inference test successful")
            print(f"   Question ID: {result['qid']}")
            print(f"   Response ({result.get('tokens', 'unknown')} tokens):")
            print(f"   '{result['answer']}'")
            return True
        else:
            print(f"❌ Inference test failed with status code {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"❌ Inference test failed: {e}")
        return False

if __name__ == "__main__":
    print("\n🔍 Running Continuous Learning Lab health check...")
    
    api_ok = check_api()
    adapter_ok = check_adapter()
    logs_ok = check_logs()
    
    if api_ok:
        inference_ok = test_inference()
    else:
        inference_ok = False
        print("❌ Skipping inference test since API is not accessible")
    
    print("\n📊 Health check summary:")
    print(f"   API Status:      {'✅' if api_ok else '❌'}")
    print(f"   Adapter Status:  {'✅' if adapter_ok else '❌'}")
    print(f"   Logs Status:     {'✅' if logs_ok else '❌'}")
    print(f"   Inference Test:  {'✅' if inference_ok else '❌'}")
    
    overall = all([api_ok, logs_ok, inference_ok])
    print(f"\n{'✅ System is healthy' if overall else '❌ System has issues'}")
    
    sys.exit(0 if overall else 1)
