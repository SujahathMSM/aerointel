# fetch_sample_docs.ps1 - download free FAA aviation documents to try real PDF ingestion.
#
# These are U.S. government works (public domain). They go in data\aviation\,
# which git ignores. The files are large; the default is the smallest useful one.
#
# Usage:
#   .\scripts\fetch_sample_docs.ps1                  # AC 43.13-1B (30 MB, 756 pages)
#   .\scripts\fetch_sample_docs.ps1 -Handbooks       # also the three AMT handbooks (about 400 MB)
#
# Then ingest:
#   python scripts\ingest_pdfs.py data\aviation --dry-run    # check how they read
#   python scripts\ingest_pdfs.py data\aviation

param(
    [switch]$Handbooks
)

$Dir = Join-Path (Split-Path -Parent $PSScriptRoot) "data\aviation"
New-Item -ItemType Directory -Force -Path $Dir | Out-Null

$Files = @(
    @{ Name = "AC_43.13-1B_CHG_1.pdf"; Url = "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_43.13-1B_CHG_1_Ed_Upd_FAA.pdf" }
)
if ($Handbooks) {
    $Files += @{ Name = "FAA-H-8083-30B_AMT_General.pdf"; Url = "https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/amtg_handbook.pdf" }
    $Files += @{ Name = "FAA-H-8083-31B_AMT_Airframe.pdf"; Url = "https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/FAA-H-8083-31B_Aviation_Maintenance_Technician_Handbook.pdf" }
    $Files += @{ Name = "FAA-H-8083-32B_AMT_Powerplant.pdf"; Url = "https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/amt_powerplant_handbook.pdf" }
}

foreach ($f in $Files) {
    $Target = Join-Path $Dir $f.Name
    if (Test-Path $Target) {
        Write-Host "already have $($f.Name)"
        continue
    }
    Write-Host "downloading $($f.Name) ..."
    curl.exe -L --fail -o $Target $f.Url
    if ($LASTEXITCODE -ne 0) {
        Write-Host "failed: $($f.Url)" -ForegroundColor Red
        if (Test-Path $Target) { Remove-Item $Target -Force }
    }
}
Write-Host "Done. Files are in $Dir"
