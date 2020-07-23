#!/usr/bin/env bash
# TODO : auto search for the serial device?
cd `dirname ${BASH_SOURCE[0]}`

gangconfig=ganglion.cfg
if [ ! -r ${gangconfig} ] ; then
    # re-create config file.
    cat <<EOF > ${gangconfig}
buffaddress=localhost:1972
# enter the Bluetoot MAC address for the ganglion below
ganglionmac=D9:5E:2A:DB:A4:0A
EOF
fi

if [ $# -gt 1 ]; then
  gangconfig=$1
  shift
fi
  
sudo bash initGanglionBluetoothSettings.sh 
while [ /bin/true ]; do
   python3 -OO pyopenBCI/utopia_ganglion.py $gangconfig $@ 
   #| tee ~/Desktop/utopia/resources/logs/ganglion.log
   sleep 5 
done
