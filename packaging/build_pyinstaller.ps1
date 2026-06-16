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
$BuildWorkDir = Join-Path $ProjectRoot "build\pyinstaller"
$DistDir = Join-Path $ProjectRoot "dist"
$InstallerDir = Join-Path $ProjectRoot "installer"
$ReleaseDir = Join-Path $ProjectRoot "release"
$ReleaseAppDir = Join-Path $ReleaseDir "AVISTA"
$Requirements = Join-Path $ProjectRoot "requirements_lock.txt"
$VersionSource = Join-Path $ProjectRoot "app\__version__.py"
$LogoPng = Join-Path $ProjectRoot "app\assets\logo.png"
$LogoIco = Join-Path $ProjectRoot "app\assets\logo.ico"
$LogoIconGenerator = Join-Path $PSScriptRoot "create_logo_icon.py"
$SpecFile = Join-Path $PSScriptRoot "avista_pyinstaller.spec"
$VersionInfoFile = Join-Path $DistDir "avista_version_info.txt"

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
$WindowsVersionTuple = ($VersionParts[0..3] -join ", ")

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

function Write-VersionInfoFile {
    param(
        [string]$Path,
        [string]$Version,
        [string]$VersionTuple,
        [string]$ApplicationName
    )

    $versionInfo = @"
# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=($VersionTuple),
    prodvers=($VersionTuple),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '040904B0',
        [
          StringStruct('CompanyName', '$ApplicationName Developers'),
          StringStruct('FileDescription', 'Automated Vehicle Infrastructure-Sensitive Tabular Analysis'),
          StringStruct('FileVersion', '$Version'),
          StringStruct('InternalName', '$ApplicationName'),
          StringStruct('OriginalFilename', '$ApplicationName.exe'),
          StringStruct('ProductName', '$ApplicationName'),
          StringStruct('ProductVersion', '$Version'),
          StringStruct('LegalCopyright', 'Copyright 2026 AVISTA Developers')
        ]
      )
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"@
    Set-Content -LiteralPath $Path -Value $versionInfo -Encoding UTF8
}

function Test-ValidIconFile {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    return (
        $bytes.Length -ge 6 -and
        $bytes[0] -eq 0 -and
        $bytes[1] -eq 0 -and
        $bytes[2] -eq 1 -and
        $bytes[3] -eq 0 -and
        ($bytes[4] -ne 0 -or $bytes[5] -ne 0)
    )
}

function New-LogoIcon {
    if (-not (Test-Path -LiteralPath $LogoPng)) {
        throw "logo.png does not exist under app\assets."
    }
    Invoke-CheckedCommand $BuildPython @(
        $LogoIconGenerator,
        $LogoPng,
        $LogoIco
    ) "Could not generate app\assets\logo.ico from logo.png."
    if (-not (Test-ValidIconFile $LogoIco)) {
        throw "Generated app\assets\logo.ico is not a valid ICO file."
    }
}

if ($Clean) {
    Remove-BuildPath $BuildEnv
    Remove-BuildPath $BuildWorkDir
    Remove-BuildPath $DistDir
    Remove-BuildPath $InstallerDir
    Remove-BuildPath $ReleaseDir
}

foreach ($directory in @($BuildWorkDir, $DistDir, $InstallerDir, $ReleaseDir)) {
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
    "-m", "pip", "install", "PyInstaller==6.17.0"
) "Could not install PyInstaller build dependency."
Invoke-CheckedCommand $BuildPython @(
    "-c",
    "import PySide6, qtawesome, torch, torchvision, torchaudio, tabpfn; print('Packaging imports verified')"
) "Required packaging imports are unavailable."

if (-not (Test-ValidIconFile $LogoIco)) {
    New-LogoIcon
}

Write-VersionInfoFile $VersionInfoFile $AppVersion $WindowsVersionTuple $AppName

$env:AVISTA_PYINSTALLER_CONSOLE = if ($Configuration -eq "Debug") { "1" } else { "0" }
$env:AVISTA_VERSION_FILE = $VersionInfoFile
$PyInstallerArgs = @(
    "-m", "PyInstaller",
    $SpecFile,
    "--noconfirm",
    "--clean",
    "--distpath", $DistDir,
    "--workpath", $BuildWorkDir
)

Push-Location $ProjectRoot
try {
    & $BuildPython @PyInstallerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
    Remove-Item Env:\AVISTA_PYINSTALLER_CONSOLE -ErrorAction SilentlyContinue
    Remove-Item Env:\AVISTA_VERSION_FILE -ErrorAction SilentlyContinue
}

$BuiltAppDir = Join-Path $DistDir $AppName
$BuiltExe = Join-Path $BuiltAppDir "$AppName.exe"
if (-not (Test-Path -LiteralPath $BuiltExe)) {
    throw "PyInstaller completed but $BuiltExe was not found."
}

Remove-BuildPath $ReleaseAppDir
New-Item -ItemType Directory -Path $ReleaseAppDir -Force | Out-Null
Copy-Item -Path (Join-Path $BuiltAppDir "*") -Destination $ReleaseAppDir -Recurse -Force

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
