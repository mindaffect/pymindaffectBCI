set batdir=%~dp0
cd %batdir%
rem utopia2ft.jar bufferhost:bufferport utopiaport utopiatimeout buffertimeout
java -Xmx64m -XX:CompressedClassSpaceSize=64m -cp UtopiaServer.jar nl.ma.utopiaserver.UtopiaServer
