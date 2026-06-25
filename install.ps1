# ReinaLook — one-line installer (Windows).
# Run:  powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/QuagKhai003/ReinaLook/main/install.ps1 | iex"
#
# Downloads the latest ReinaLook.exe from GitHub Releases into %LOCALAPPDATA%\ReinaLook,
# creates Start-Menu + Desktop shortcuts, and launches the app. No admin rights needed.

$ErrorActionPreference = 'Stop'
$Repo    = 'QuagKhai003/ReinaLook'
$AppName = 'ReinaLook'
$InstallDir = Join-Path $env:LOCALAPPDATA $AppName
$ExePath    = Join-Path $InstallDir 'ReinaLook.exe'

Write-Host "Installing $AppName..." -ForegroundColor Cyan

# 1) find the ReinaLook.exe asset on the latest GitHub Release
Write-Host "  finding the latest release..."
$headers = @{ 'User-Agent' = 'ReinaLook-Installer' }
$rel = Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/$Repo/releases/latest"
$asset = $rel.assets | Where-Object { $_.name -eq 'ReinaLook.exe' } | Select-Object -First 1
if (-not $asset) { throw "No 'ReinaLook.exe' asset found on the latest release of $Repo." }
Write-Host "  release $($rel.tag_name)  ($([math]::Round($asset.size/1MB,1)) MB)"

# 2) close any running instance so an update can overwrite the exe, then download
Get-Process -Name 'ReinaLook' -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Write-Host "  downloading to $ExePath ..."
Invoke-WebRequest -Headers $headers -Uri $asset.browser_download_url -OutFile $ExePath

# 3) shortcuts (Start Menu + Desktop)
$ws = New-Object -ComObject WScript.Shell
foreach ($dir in @(
    (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'),
    [Environment]::GetFolderPath('Desktop')
)) {
    $lnk = $ws.CreateShortcut((Join-Path $dir "$AppName.lnk"))
    $lnk.TargetPath = $ExePath
    $lnk.WorkingDirectory = $InstallDir
    $lnk.Description = 'ReinaLook — LUT generator'
    $lnk.Save()
}

# 4) record install location for the uninstaller
Set-Content -Path (Join-Path $InstallDir 'install.json') -Value (@{
    name = $AppName; version = $rel.tag_name; installed = (Get-Date -Format o); exe = $ExePath
} | ConvertTo-Json)

Write-Host "Done. Launching $AppName..." -ForegroundColor Green
Write-Host "  (Start Menu + Desktop shortcut created. Uninstall: see README.)" -ForegroundColor DarkGray
Start-Process $ExePath
