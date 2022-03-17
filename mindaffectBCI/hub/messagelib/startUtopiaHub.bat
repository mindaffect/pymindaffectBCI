set batdir=%~dp0
cd %batdir%
java -jar UtopiaServer.jar 8400 -1
rem  | ..\..\resources\bin\mtee /t ..\..\resources\logs\utopiahub_%uid%.log
