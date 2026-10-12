# try_api.ps1 - easy way to poke the AeroIntel API from PowerShell.
# Usage:
#   .\scripts\try_api.ps1 ingest "If brake temp exceeds 300C, cool before departure."
#   .\scripts\try_api.ps1 query  "what should I check if hydraulic pressure drops?"
#   .\scripts\try_api.ps1 search "engine shaking while ascending"
#   .\scripts\try_api.ps1 stream "what should I check if hydraulic pressure drops?"
#   .\scripts\try_api.ps1 health
#
# /v1 calls need a token. Set these once per shell (a client from AUTH_CLIENTS in .env):
#   $env:API_CLIENT_ID = "admin"; $env:API_CLIENT_SECRET = "<its secret>"

param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Action,
    [Parameter(Position = 1)][string]$Text
)

$Base = "http://localhost:8000"

function Get-Token() {
    if (-not $env:API_CLIENT_ID -or -not $env:API_CLIENT_SECRET) {
        Write-Host "Set `$env:API_CLIENT_ID and `$env:API_CLIENT_SECRET first (see the top of this script)." -ForegroundColor Red
        exit 1
    }
    $file = New-TemporaryFile
    @{ client_id = $env:API_CLIENT_ID; client_secret = $env:API_CLIENT_SECRET } | ConvertTo-Json | Set-Content -Path $file -Encoding UTF8
    try {
        $raw = curl.exe -s -X POST "$Base/auth/token" -H "Content-Type: application/json" -d "@$file"
    } finally {
        Remove-Item $file -Force
    }
    $token = ($raw | ConvertFrom-Json).access_token
    if (-not $token) { Write-Host "Could not get a token: $raw" -ForegroundColor Red; exit 1 }
    return $token
}

function Send-Json($method, $path, $body) {
    $token = Get-Token
    $file = New-TemporaryFile
    $body | Set-Content -Path $file -Encoding UTF8
    try {
        curl.exe -s -X $method "$Base$path" -H "Content-Type: application/json" -H "Authorization: Bearer $token" -d "@$file"
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
    "stream" {
        if (-not $Text) { Write-Host "Usage: .\scripts\try_api.ps1 stream `<question`>" -ForegroundColor Red; exit 1 }
        # -N turns off curl's output buffering so events print as they arrive.
        $token = Get-Token
        $file = New-TemporaryFile
        (@{ question = $Text } | ConvertTo-Json) | Set-Content -Path $file -Encoding UTF8
        try {
            curl.exe -s -N -X POST "$Base/v1/query/stream" -H "Content-Type: application/json" -H "Authorization: Bearer $token" -d "@$file"
        } finally {
            Remove-Item $file -Force
        }
        Write-Host ""
    }
    default {
        Write-Host "Unknown action '$Action'. Use: health | ingest | query | search | stream" -ForegroundColor Red
        exit 1
    }
}