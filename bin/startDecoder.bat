setlocal enabledelayedexpansion
set batdir=%~dp0
cd %batdir%

python3 -m mindaffectBCI.decoder.decoder %1 %2 %3 %4 %5
