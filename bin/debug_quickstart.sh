#!/bin/bash
cd `dirname ${BASH_SOURCE[0]}`
buffdir=`dirname $0`

python3 -m mindaffectBCI.online_bci --config_file fakedata.json