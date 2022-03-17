#!/bin/bash
# start a test client
cd `dirname ${BASH_SOURCE[0]}`

# start the server, in the background -- if can
#java -jar UtopiaServer.jar 8400 1 &
#serverpid=$!
#echo ServerPID $serverpid
#sleep 2
## kill the server on ctrl-c
#killserver() {
#    kill $serverpid
#}
#trap 'killserver' SIGINT

java -cp UtopiaServer.jar nl.ma.utopiaserver.UtopiaClient $@
#killserver
