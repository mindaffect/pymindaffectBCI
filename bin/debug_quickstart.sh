#!/bin/bash
cd `dirname ${BASH_SOURCE[0]}`
buffdir=`dirname $0`

echo Starting the utopia-hub
bash startUtopiaHub.sh &
utopia2ftpid=$!
echo utopia2ftpid=$utopia2ftpid
sleep 5

#resources/eeg/startJavaSignalproxy.sh > /dev/null 
./startUtopiaClient.sh data 200

