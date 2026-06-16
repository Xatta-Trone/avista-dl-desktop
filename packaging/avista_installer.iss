#ifndef MyAppName
  #error MyAppName must be supplied by build_pyinstaller.ps1 from app/__version__.py
#endif
#ifndef MyAppVersion
  #error MyAppVersion must be supplied by build_pyinstaller.ps1 from app/__version__.py
#endif
#define MyAppPublisher "AVISTA Developers"
#define MyAppExeName "AVISTA.exe"

[Setup]
AppId={{B151A258-8635-4D51-8AD7-E83D99D7D272}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\AVISTA
DefaultGroupName=AVISTA
DisableProgramGroupPage=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\installer
OutputBaseFilename=AVISTA_Setup
SetupIconFile=..\app\assets\logo.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
LicenseFile=..\LICENSE.txt
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ChangesAssociations=yes
CloseApplications=yes
RestartApplications=no

[Files]
Source: "..\release\AVISTA\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\AVISTA"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\AVISTA"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: checkedonce

[Registry]
Root: HKA; Subkey: "Software\Classes\.avista"; ValueType: string; ValueName: ""; ValueData: "AVISTA.Project"; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.avista\OpenWithProgids"; ValueType: string; ValueName: "AVISTA.Project"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\AVISTA.Project"; ValueType: string; ValueName: ""; ValueData: "AVISTA Project"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\AVISTA.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKA; Subkey: "Software\Classes\AVISTA.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch AVISTA"; Flags: nowait postinstall skipifsilent
