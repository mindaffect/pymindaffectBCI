#!/bin/bash
cd `dirname ${BASH_SOURCE[0]}`
cd ../mindaffectBCI/hub
uid=`date +%y%m%d_%H%M`

# start up the utopia-hub
echo Starting the Utopia-HUB
if [ ! -z `which xterm` ]; then
    # start in new xterm
    xterm -iconic -title "Utopia-HUB" java -jar UtopiaServer.jar 8400 0 ../../logs/mindaffectBCI.txt
else # run directly
   java -jar UtopiaServer.jar 8400 0 ../../logs/mindaffectBCI.txt 
fi
