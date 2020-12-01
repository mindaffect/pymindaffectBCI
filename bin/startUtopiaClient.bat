set batdir=%~dp0
cd %batdir%
cd ../mindaffectBCI/hub
rem utopia2ft.jar bufferhost:bufferport utopiaport utopiatimeout buffertimeout
java -Xmx64m -XX:CompressedClassSpaceSize=64m -cp UtopiaServer.jar nl.ma.utopiaserver.UtopiaClient %1 %2 %3 %4
