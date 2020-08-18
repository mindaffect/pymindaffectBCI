set batdir=%~dp0
cd %batdir%
mkdir ..\logs
java -jar UtopiaServer.jar 8400 0 ..\logs\mindaffectBCI.txt
