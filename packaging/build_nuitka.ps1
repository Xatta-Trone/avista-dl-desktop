[CmdletBinding()]
param(
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [string]$PythonExecutable = "python",
    [switch]$Clean,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildEnv = Join-Path $ProjectRoot "build_env"
$BuildPython = Join-Path $BuildEnv "Scripts\python.exe"
$DistDir = Join-Path $ProjectRoot "dist"
$InstallerDir = Join-Path $ProjectRoot "installer"
$ReleaseDir = Join-Path $ProjectRoot "release"
$ReleaseAppDir = Join-Path $ReleaseDir "AVISTA"
$Requirements = Join-Path $ProjectRoot "requirements_lock.txt"
$VersionSource = Join-Path $ProjectRoot "app\__version__.py"
$LogoPng = Join-Path $ProjectRoot "app\assets\logo.png"
$LogoIco = Join-Path $ProjectRoot "app\assets\logo.ico"

$VersionText = Get-Content -LiteralPath $VersionSource -Raw
$VersionMatch = [regex]::Match($VersionText, '(?m)^__version__\s*=\s*"([^"]+)"')
$NameMatch = [regex]::Match($VersionText, '(?m)^APP_NAME\s*=\s*"([^"]+)"')
if (-not $VersionMatch.Success -or -not $NameMatch.Success) {
    throw "Could not read APP_NAME and __version__ from $VersionSource."
}
$AppVersion = $VersionMatch.Groups[1].Value
$AppName = $NameMatch.Groups[1].Value
$VersionParts = @($AppVersion.Split("."))
while ($VersionParts.Count -lt 4) {
    $VersionParts += "0"
}
$WindowsVersion = ($VersionParts[0..3] -join ".")

function Invoke-CheckedCommand {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$FailureMessage
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage Exit code: $LASTEXITCODE."
    }
}

function Remove-BuildPath {
    param([string]$Path)

    $resolvedRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
    $resolvedTarget = [System.IO.Path]::GetFullPath($Path)
    if (-not $resolvedTarget.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove path outside the project root: $resolvedTarget"
    }
    if (Test-Path -LiteralPath $resolvedTarget) {
        Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
    }
}

if ($Clean) {
    Remove-BuildPath $BuildEnv
    Remove-BuildPath $DistDir
    Remove-BuildPath $InstallerDir
    Remove-BuildPath $ReleaseDir
}

foreach ($directory in @($DistDir, $InstallerDir, $ReleaseDir)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

if (-not (Test-Path -LiteralPath $BuildPython)) {
    Invoke-CheckedCommand $PythonExecutable @(
        "-m", "venv", $BuildEnv
    ) "Could not create build_env with $PythonExecutable."
}

Invoke-CheckedCommand $BuildPython @(
    "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"
) "Could not upgrade build tooling."
Invoke-CheckedCommand $BuildPython @(
    "-m", "pip", "install", "-r", $Requirements
) "Could not install requirements_lock.txt."
Invoke-CheckedCommand $BuildPython @(
    "-m", "pip", "install", "Nuitka", "ordered-set", "zstandard"
) "Could not install Nuitka build dependencies."
Invoke-CheckedCommand $BuildPython @(
    "-c",
    "import PySide6, qtawesome, torch, torchvision, torchaudio, tabpfn; print('Packaging imports verified')"
) "Required packaging imports are unavailable."

if (-not (Test-Path -LiteralPath $LogoIco)) {
    if (-not (Test-Path -LiteralPath $LogoPng)) {
        throw "Neither logo.ico nor logo.png exists under app\assets."
    }
    Invoke-CheckedCommand $BuildPython @(
        "-c",
        "from PIL import Image; Image.open(r'$LogoPng').convert('RGBA').save(r'$LogoIco', sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])"
    ) "Could not generate app\assets\logo.ico from logo.png."
    if (-not (Test-Path -LiteralPath $LogoIco)) {
        throw "Could not generate app\assets\logo.ico from logo.png."
    }
}

$ConsoleMode = if ($Configuration -eq "Debug") { "force" } else { "disable" }
$NuitkaArgs = @(
    "-m", "nuitka",
    (Join-Path $ProjectRoot "main.py"),
    "--mode=standalone",
    "--enable-plugin=pyside6",
    "--windows-console-mode=$ConsoleMode",
    "--windows-icon-from-ico=$LogoIco",
    "--include-data-dir=$(Join-Path $ProjectRoot 'app\assets')=app/assets",
    "--module-parameter=torch-disable-jit=no",
    "--include-package=app",
    "--include-package=torch",
    "--include-package=torchvision",
    "--include-package=torchaudio",
    "--include-package=tabpfn",
    "--include-package=xgboost",
    "--include-package=lightgbm",
    "--include-package=sklearn",
    "--include-package=imblearn",
    "--include-package=matplotlib",
    "--include-package-data=qtawesome",
    "--include-package-data=matplotlib",
    "--include-package-data=tabpfn",
    "--output-dir=$DistDir",
    "--output-filename=$AppName.exe",
    "--company-name=$AppName",
    "--product-name=$AppName",
    "--file-description=Automated Vehicle Infrastructure-Sensitive Tabular Analysis",
    "--file-version=$WindowsVersion",
    "--product-version=$WindowsVersion",
    "--copyright=Copyright 2026 AVISTA Developers",
    "--msvc=latest",
    "--assume-yes-for-downloads",
    "--report=$(Join-Path $DistDir 'nuitka-compilation-report.xml')",
    "--report-template=LicenseReport:$(Join-Path $DistDir 'nuitka-license-report.txt')"
)

Push-Location $ProjectRoot
try {
    & $BuildPython @NuitkaArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Nuitka build failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

$BuiltExe = Get-ChildItem -Path $DistDir -Recurse -Filter "AVISTA.exe" |
    Select-Object -First 1
if (-not $BuiltExe) {
    throw "Nuitka completed but AVISTA.exe was not found under $DistDir."
}

Remove-BuildPath $ReleaseAppDir
New-Item -ItemType Directory -Path $ReleaseAppDir -Force | Out-Null
Copy-Item -Path (Join-Path $BuiltExe.Directory.FullName "*") -Destination $ReleaseAppDir -Recurse -Force

foreach ($document in @(
    "LICENSE.txt",
    "THIRD_PARTY_NOTICES.txt",
    "README.md",
    "DEVELOPER_GUIDE.md",
    "CHANGELOG.md"
)) {
    $source = Join-Path $ProjectRoot $document
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Required release document is missing: $source"
    }
    Copy-Item -LiteralPath $source -Destination $ReleaseAppDir -Force
}

if (-not $SkipInstaller) {
    $IsccCandidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
    )
    $Iscc = $IsccCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } |
        Select-Object -First 1
    if (-not $Iscc) {
        throw "Inno Setup 6 was not found. Install it or rerun with -SkipInstaller."
    }

    & $Iscc "/DMyAppName=$AppName" "/DMyAppVersion=$AppVersion" `
        (Join-Path $PSScriptRoot "avista_installer.iss")
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup failed with exit code $LASTEXITCODE."
    }

    $InstallerPath = Join-Path $InstallerDir "AVISTA_Setup.exe"
    if (-not (Test-Path -LiteralPath $InstallerPath)) {
        throw "Inno Setup completed but $InstallerPath was not created."
    }
    Copy-Item -LiteralPath $InstallerPath -Destination $ReleaseDir -Force
}

Write-Host "AVISTA standalone build: $ReleaseAppDir"
if (-not $SkipInstaller) {
    Write-Host "AVISTA installer output: $InstallerDir"
}
