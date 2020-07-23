setlocal enabledelayedexpansion
set batdir=%~dp0
cd %batdir%

echo Starting the Utopia-HUB
start startUtopiaHub.bat

rem Weird windows hack to sleep for 2 secs to allow the buffer server to start
ping 127.0.0.1 -n 3 > nul

echo Starting the data acquisation device %dataacq% \(background this shell\)
startUtopiaClient.bat data 200
