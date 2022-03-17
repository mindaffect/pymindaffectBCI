#!/bin/bash
# start a test client
cd `dirname ${BASH_SOURCE[0]}`
java -cp UtopiaServer.jar nl.ma.utopiaserver.SSDP $@
