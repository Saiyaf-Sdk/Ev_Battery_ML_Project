# Interactive launcher for EV Battery Machine Learning Project
$ErrorActionPreference = "Stop"

# Ensure virtual environment is activated
if (-not (Get-Command "python" -ErrorAction SilentlyContinue) -or -not ($env:VIRTUAL_ENV)) {
    if (Test-Path ".\.venv\Scripts\Activate.ps1") {
        Write-Host "Activating virtual environment..." -ForegroundColor Cyan
        & ".\.venv\Scripts\Activate.ps1"
    }
}

# Run the interactive console dashboard
python -m evbattery.ui

