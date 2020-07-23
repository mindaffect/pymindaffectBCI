#!/bin/bash
cd `dirname ${BASH_SOURCE[0]}`
buffdir=`dirname ${BASH_SOURCE[0]}`
echo Starting the java buffer server \(background\)
echo pwd

bash ${buffdir}/startUtopiaHub.sh & 
bufferpid=$!
echo buffpid=$bufferpid
sleep 5

bash ../mindaffectBCI/examples/acquisation/startGanglion.sh 

