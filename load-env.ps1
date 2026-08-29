# load-env.ps1
# Loads variables from .env into the current PowerShell session.
# Needed because Python doesn't read .env automatically — only docker compose does.

if (-not (Test-Path .env)) {
    Write-Host "No .env file found in current directory." -ForegroundColor Red
    exit 1
}

Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]*)=(.*)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        [System.Environment]::SetEnvironmentVariable($name, $value)
        Write-Host "Set $name" -ForegroundColor Green
    }
}