#!/usr/bin/env python3
"""
Script to test the authentication endpoint
"""

import requests
import json
from urllib3.exceptions import InsecureRequestWarning
import urllib3

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(InsecureRequestWarning)

def test_auth_endpoint():
    try:
        # Test the auth endpoint with a mock request
        url = "https://localhost:8443/auth/login"
        
        # Just test if the endpoint exists (will return 422 for invalid data, but that's expected)
        response = requests.post(url, json={"id_token": "test"}, verify=False, timeout=5)
        
        print(f"Auth endpoint status: {response.status_code}")
        
        if response.status_code == 422:
            print("✅ Auth endpoint is accessible (returns 422 for invalid token, which is expected)")
            print(f"Response: {response.text}")
            return True
        elif response.status_code == 401:
            print("✅ Auth endpoint is accessible (returns 401 for invalid token, which is expected)")
            print(f"Response: {response.text}")
            return True
        else:
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to auth endpoint")
        return False
    except Exception as e:
        print(f"❌ Error testing auth endpoint: {e}")
        return False

if __name__ == "__main__":
    print("Testing Auth Endpoint...")
    test_auth_endpoint()
