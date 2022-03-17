#!/bin/bash
# start a test client
cd `dirname ${BASH_SOURCE[0]}`

nclients=5

# start the server, in the background -- if can
echo Starting Utopia-HUB
java -Xmx128m -jar UtopiaServer.jar 8400 1 &
serverpid=$!
echo ServerPID $serverpid
sleep 2
# kill the server on ctrl-c
killserver() {
    kill $serverpid
}
trap 'killserver' SIGINT

declare -a clientpid
for i in {1..50}; do
    echo Starting client $i
    java -cp UtopiaServer.jar nl.ma.utopiaserver.UtopiaClient $@ > /dev/null 2>&1 &
    clientpid[$i]=$!
    sleep 1
done
echo $clientpid

declare -a newclientpid
for i in {1..50}; do
    # sleep random time, then kill client
    sleep $(($RANDOM%10))
    echo killing client $i = ${clientpid[$i]}
    kill ${clientpid[$i]}
    # spawn new client
    echo Starting new client $i
    java -cp UtopiaServer.jar nl.ma.utopiaserver.UtopiaClient $@ > /dev/null 2>&1 &
    newclientpid[$i]=$!    
done

# kill everything
echo Killing all clients
for i in "$(newclientpid)"; do
    kill ${newclientpid[i]}
done
killserver
