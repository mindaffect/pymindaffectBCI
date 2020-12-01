#!/bin/bash
# start a test client
cd `dirname ${BASH_SOURCE[0]}`
cd ../mindaffectBCI/hub

java -cp UtopiaServer.jar nl.ma.utopiaserver.UtopiaClient $@
