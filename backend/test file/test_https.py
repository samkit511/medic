#!/usr/bin/env python3
"""
Script to test the HTTPS backend endpoint
"""

import requests
import json
from urllib3.exceptions import InsecureRequestWarning
import urllib3

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(InsecureRequestWarning)

def test_backend():
    try:
        # Test the root endpoint
        response = requests.get("https://localhost:8443/", verify=False, timeout=5)
        
        if response.status_code == 200:
            print("✅ Backend HTTPS endpoint is working!")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Backend returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Make sure it's running on https://localhost:8443")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing backend: {e}")
        return False

if __name__ == "__main__":
    print("Testing HTTPS Backend Connection...")
    test_backend()
