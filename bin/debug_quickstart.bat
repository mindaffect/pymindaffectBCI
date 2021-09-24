setlocal enabledelayedexpansion
set batdir=%~dp0
cd %batdir%

rem search for the right python
for %%i in (python.exe, python3.exe) do set pyexe=%%~$PATH:i

%pyexe% -m mindaffectBCI.online_bci --config_file fakedata.json