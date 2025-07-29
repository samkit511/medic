#!/usr/bin/env powershell
Write-Host "Starting CURA Medical Chatbot Backend..." -ForegroundColor Green
Write-Host "Server will be available at: https://localhost:8443" -ForegroundColor Yellow
Write-Host "API Documentation: https://localhost:8443/docs" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host ""

# Change to the backend directory
Set-Location "C:\Users\samkit jain\Dropbox\PC\Desktop\research\landing\medic\backend"

# Start the Python server
python main_simple.py
