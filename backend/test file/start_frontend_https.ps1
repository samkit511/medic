# PowerShell script to start React frontend with HTTPS support
# This script configures Vite to serve the frontend over HTTPS

Write-Host "🚀 Starting Medical Chatbot Frontend with HTTPS..." -ForegroundColor Green
Write-Host "📍 Frontend will be available at: https://localhost:5173" -ForegroundColor Cyan
Write-Host "🔒 SSL/TLS encryption: ENABLED" -ForegroundColor Green
Write-Host ""
Write-Host "⚠️  Note: You may see a security warning in your browser" -ForegroundColor Yellow
Write-Host "   Click 'Advanced' -> 'Proceed to localhost (unsafe)' to continue" -ForegroundColor Yellow
Write-Host ""
Write-Host "🛑 Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host ""

# Change to the project root directory
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Check if package.json exists
if (-not (Test-Path "package.json")) {
    Write-Host "❌ package.json not found! Make sure you're in the correct directory." -ForegroundColor Red
    exit 1
}

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install dependencies!" -ForegroundColor Red
        exit 1
    }
}

# Start the Vite dev server with HTTPS
try {
    # Set environment variable for HTTPS
    $env:HTTPS = "true"
    
    # Start the development server
    Write-Host "🌐 Starting Vite development server..." -ForegroundColor Green
    npm run dev -- --https
}
catch {
    Write-Host "❌ Error starting frontend server: $_" -ForegroundColor Red
    exit 1
}
