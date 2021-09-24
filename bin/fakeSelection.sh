#!/bin/bash
cd `dirname ${BASH_SOURCE[0]}`
cd ../mindaffectBCI/hub

java -cp UtopiaServer.jar nl.ma.utopiaserver.UtopiaClient spawnserver selection
