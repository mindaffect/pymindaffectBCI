#! /usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`
buffdir="$( pwd )"
echo $buffdir
exedir=${buffdir}/eego2ft
buffexe=eego2utopia;
if [ `uname -s` == 'Linux' ]; then
   if [[ `uname -m` =~ arm* ]]; then
      exedir=$buffdir'/../mindaffectBCI/examples/acquisation/eego2ft/linux/arm32bit';
   else # x86 linux 
	  exedir=$buffdir'/../mindaffectBCI/examples/acquisation/eego2ft/linux/32bit';
   fi
   export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${exedir}
else # Mac
   exedir=$buffdir/../examples/acquisation/eego2ft/maci
   # add exec directory to library load path
   export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:${exedir}   
fi

while [ true ] ; do 
  ${exedir}/${buffexe} 
  sleep 1
done

if [ $? == 1 ] ; then
	 echo Couldnt start the AMP driver.  Possible reasons
	 echo 1\) The amplifier isnt connected or turned on?
	 echo 2\) You cannot read the USB device.  On linux try: sudo ./${BASH_SOURCE[0]}
fi
