# try_api.ps1 - easy way to poke the AeroIntel API from PowerShell.
# Usage:
#   .\scripts\try_api.ps1 ingest "If brake temp exceeds 300C, cool before departure."
#   .\scripts\try_api.ps1 query  "what should I check if hydraulic pressure drops?"
#   .\scripts\try_api.ps1 search "engine shaking while ascending"
#   .\scripts\try_api.ps1 health

param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Action,
    [Parameter(Position = 1)][string]$Text
)

$Base = "http://localhost:8000"

function Send-Json($method, $path, $body) {
    $file = New-TemporaryFile
    $body | Set-Content -Path $file -Encoding UTF8
    try {
        curl.exe -s -X $method "$Base$path" -H "Content-Type: application/json" -d "@$file"
    } finally {
        Remove-Item $file -Force
    }
    Write-Host ""
}

switch ($Action) {
    "health" {
        Write-Host "live:  $(curl.exe -s $Base/health/live)"
        Write-Host "ready: $(curl.exe -s $Base/health/ready)"
    }
    "ingest" {
        if (-not $Text) { Write-Host "Usage: .\scripts\try_api.ps1 ingest `<text`>" -ForegroundColor Red; exit 1 }
        $body = @{ content = $Text; metadata = @{ source = "manual_test" } } | ConvertTo-Json
        Send-Json "POST" "/v1/ingest" $body
    }
    "query" {
        if (-not $Text) { Write-Host "Usage: .\scripts\try_api.ps1 query `<question`>" -ForegroundColor Red; exit 1 }
        $body = @{ question = $Text } | ConvertTo-Json
        Send-Json "POST" "/v1/query" $body
    }
    "search" {
        if (-not $Text) { Write-Host "Usage: .\scripts\try_api.ps1 search `<query`>" -ForegroundColor Red; exit 1 }
        $body = @{ query = $Text } | ConvertTo-Json
        Send-Json "POST" "/v1/search" $body
    }
    default {
        Write-Host "Unknown action '$Action'. Use: health | ingest | query | search" -ForegroundColor Red
        exit 1
    }
}