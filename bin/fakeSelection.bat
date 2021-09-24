setlocal enabledelayedexpansion
set batdir=%~dp0
cd %batdir%
cd ..\mindaffectBCI\hub

java -cp UtopiaServer.jar nl.ma.utopiaserver.UtopiaClient selection
