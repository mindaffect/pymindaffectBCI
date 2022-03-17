#!/bin/bash
cd `dirname ${BASH_SOURCE[0]}`
# start the server, in the background
java -jar UtopiaServer.jar 8400 2 &
serverpid=$!
echo ServerPID $serverpid
# start the listening client
sleep .5
java -cp UtopiaServer.jar nl.ma.utopiaserver.UtopiaClient &
listeningpid=$!
# start sending client
sleep 1
java -cp UtopiaServer.jar nl.ma.utopiaserver.UtopiaClient 10 &
sendingpid=$!

# kill the server+clients on ctrl-c
killserver() {
    kill $serverpid
    kill $sendingpid
    kill $listeningpid
}
trap 'killserver' SIGINT

sleep 30
killserver
