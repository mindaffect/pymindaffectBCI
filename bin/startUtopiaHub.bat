set batdir=%~dp0
cd %batdir%
cd ../mindaffectBCI/hub
java -jar UtopiaServer.jar 8400 0 ..\..\logs\mindaffectBCI.txt
