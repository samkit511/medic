#!/usr/bin/env python3
"""
Script to convert PFX certificate to PEM format for uvicorn SSL configuration
"""

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import pkcs12
import os

def convert_pfx_to_pem():
    pfx_path = "certs/localhost.pfx"
    cert_path = "certs/localhost.crt"
    key_path = "certs/localhost.key"
    password = b"localhost123"
    
    try:
        # Read the PFX file
        with open(pfx_path, "rb") as pfx_file:
            pfx_data = pfx_file.read()
        
        # Load the PFX
        private_key, certificate, additional_certificates = pkcs12.load_key_and_certificates(
            pfx_data, password
        )
        
        # Write the certificate to PEM format
        with open(cert_path, "wb") as cert_file:
            cert_file.write(certificate.public_bytes(serialization.Encoding.PEM))
        
        # Write the private key to PEM format
        with open(key_path, "wb") as key_file:
            key_file.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        print(f"Certificate converted successfully!")
        print(f"Certificate file: {cert_path}")
        print(f"Private key file: {key_path}")
        
        return True
        
    except Exception as e:
        print(f"Error converting certificate: {e}")
        return False

if __name__ == "__main__":
    convert_pfx_to_pem()
