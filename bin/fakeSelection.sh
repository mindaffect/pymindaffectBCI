#!/bin/bash
cd `dirname ${BASH_SOURCE[0]}`
java -cp UtopiaServer.jar nl.ma.utopiaserver.UtopiaClient spawnserver selection
