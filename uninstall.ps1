# ReinaLook — uninstaller (Windows). Removes EVERYTHING the app installed.
# Run:  powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/QuagKhai003/ReinaLook/main/uninstall.ps1 | iex"

$ErrorActionPreference = 'SilentlyContinue'
$AppName    = 'ReinaLook'
$InstallDir = Join-Path $env:LOCALAPPDATA $AppName

Write-Host "Uninstalling $AppName..." -ForegroundColor Cyan

# 1) stop the app if it is running
Get-Process -Name $AppName -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

# 2) remove shortcuts (Start Menu + Desktop)
$shortcuts = @(
    (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\$AppName.lnk"),
    (Join-Path ([Environment]::GetFolderPath('Desktop')) "$AppName.lnk")
)
foreach ($s in $shortcuts) { if (Test-Path $s) { Remove-Item $s -Force; Write-Host "  removed shortcut: $s" } }

# 3) remove the install folder (the exe + install record)
if (Test-Path $InstallDir) {
    Remove-Item $InstallDir -Recurse -Force
    Write-Host "  removed: $InstallDir"
}

# 4) remove the PyInstaller temp-extract cache it may have left (_MEIxxxx) for this app
Get-ChildItem $env:TEMP -Directory -Filter '_MEI*' -ErrorAction SilentlyContinue | ForEach-Object {
    if (Test-Path (Join-Path $_.FullName 'lutgen')) { Remove-Item $_.FullName -Recurse -Force }
}

Write-Host "$AppName fully removed. (Your exported .cube/.json files are untouched.)" -ForegroundColor Green
