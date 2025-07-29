#!/usr/bin/env python3
"""
Quick test to identify which backend is running
"""

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    response = requests.get("https://localhost:8443/", verify=False, timeout=5)
    if response.status_code == 200:
        data = response.json()
        print("Backend Response:")
        print(f"Message: {data.get('message')}")
        print(f"Version: {data.get('version')}")
        print("Available endpoints:")
        endpoints = data.get('endpoints', {})
        for endpoint, description in endpoints.items():
            print(f"  {endpoint}: {description}")
        
        # Check if this is the simple or complex backend
        if len(endpoints) <= 3:
            print("\n✅ This appears to be the SIMPLE backend")
        else:
            print("\n⚠️ This appears to be the COMPLEX backend (might cause timeouts)")
            
except Exception as e:
    print(f"Error: {e}")
