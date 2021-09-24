#!/bin/bash
cd `dirname ${BASH_SOURCE[0]}`

spawnhub=1
verb=1
java -cp "FakeRecogniser.jar:UtopiaServer.jar" nl.ma.utopia.fakerecogniser.FakeRecogniser localhost:8400 $spawnhub $verb
