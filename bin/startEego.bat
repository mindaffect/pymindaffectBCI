setlocal enabledelayedexpansion
set batdir=%~dp0
cd %batdir%

rem You may need to change the com-port!
cd ../mindaffectBCI/examples/acquisation
eego2utopia.exe
