# PowerShell script to trust the self-signed SSL certificate
# This helps avoid browser security warnings for local development

param(
    [switch]$Remove = $false
)

# Check if running as administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

$certificatePath = ".\backend\certs\localhost.crt"

if ($Remove) {
    Write-Host "🗑️  Removing trusted certificate..." -ForegroundColor Yellow
    
    try {
        # Remove from Trusted Root Certification Authorities
        $cert = Get-ChildItem -Path "Cert:\LocalMachine\Root" | Where-Object { $_.Subject -like "*localhost*" -and $_.Issuer -like "*localhost*" }
        if ($cert) {
            $cert | Remove-Item
            Write-Host "✅ Certificate removed from Trusted Root Certification Authorities" -ForegroundColor Green
        } else {
            Write-Host "ℹ️  No matching certificate found in Trusted Root" -ForegroundColor Blue
        }
        
        # Remove from Personal store
        $cert = Get-ChildItem -Path "Cert:\LocalMachine\My" | Where-Object { $_.Subject -like "*localhost*" -and $_.Issuer -like "*localhost*" }
        if ($cert) {
            $cert | Remove-Item
            Write-Host "✅ Certificate removed from Personal store" -ForegroundColor Green
        } else {
            Write-Host "ℹ️  No matching certificate found in Personal store" -ForegroundColor Blue
        }
    }
    catch {
        Write-Host "❌ Error removing certificate: $_" -ForegroundColor Red
    }
    
    Read-Host "Press Enter to exit"
    exit 0
}

# Check if certificate file exists
if (-not (Test-Path $certificatePath)) {
    Write-Host "❌ Certificate file not found: $certificatePath" -ForegroundColor Red
    Write-Host "Run 'python backend\generate_ssl.py' first to create SSL certificates." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "🔒 Medical Chatbot Certificate Trust Utility" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green
Write-Host ""
Write-Host "This script will add the self-signed SSL certificate to Windows" -ForegroundColor White
Write-Host "Trusted Root Certification Authorities to avoid browser warnings." -ForegroundColor White
Write-Host ""
Write-Host "Certificate file: $certificatePath" -ForegroundColor Cyan
Write-Host ""

$response = Read-Host "Do you want to trust this certificate? (y/N)"
if ($response -ne 'y' -and $response -ne 'Y') {
    Write-Host "❌ Certificate not trusted. Exiting..." -ForegroundColor Red
    exit 0
}

try {
    Write-Host "📋 Adding certificate to Trusted Root Certification Authorities..." -ForegroundColor Yellow
    
    # Import the certificate to Trusted Root Certification Authorities
    Import-Certificate -FilePath $certificatePath -CertStoreLocation "Cert:\LocalMachine\Root" -Confirm:$false | Out-Null
    
    Write-Host "✅ Certificate successfully added to Trusted Root!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 You can now access your application without browser warnings:" -ForegroundColor Green
    Write-Host "   Backend:  https://localhost:8000" -ForegroundColor Cyan
    Write-Host "   Frontend: https://localhost:5173" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🗑️  To remove the certificate later, run:" -ForegroundColor Yellow
    Write-Host "   .\trust_certificate.ps1 -Remove" -ForegroundColor Gray
}
catch {
    Write-Host "❌ Error trusting certificate: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Alternative: You can manually add the certificate:" -ForegroundColor Yellow
    Write-Host "1. Open 'certmgr.msc' (Certificate Manager)" -ForegroundColor Gray
    Write-Host "2. Navigate to Trusted Root Certification Authorities > Certificates" -ForegroundColor Gray
    Write-Host "3. Right-click > All Tasks > Import" -ForegroundColor Gray
    Write-Host "4. Select the certificate file: $certificatePath" -ForegroundColor Gray
}

Write-Host ""
Read-Host "Press Enter to exit"
