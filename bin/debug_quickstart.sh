#!/bin/bash
cd `dirname ${BASH_SOURCE[0]}`
buffdir=`dirname $0`

echo Starting the utopia-hub
java/messagelib/startUtopiaHub.sh &
utopia2ftpid=$!
echo utopia2ftpid=$utopia2ftpid
sleep 5

#resources/eeg/startJavaSignalproxy.sh > /dev/null 
java/messagelib/testclient.sh data
dataacqpid=$!
echo dataacqpid=$dataacqpid

echo Starting the event viewer
#bash resources/eeg/startJavaEventViewer.sh
#bash matlab/utopia/startRecogniser.sh 
kill $utopia2ftpid
kill $dataacqpid
#kill $bufferpid
