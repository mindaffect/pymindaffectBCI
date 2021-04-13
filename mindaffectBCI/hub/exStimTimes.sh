#!/bin/bash
logfile=UtopiaMessages.log
if [ $# -gt 0 ]; then
	logfile=$1
fi
echo FlipTimes; sed -n '/LOG.*Flip:/p' $logfile | sed 's/.*\sts:\([-0-9]*\)\s.*Flip:\([-0-9]*\)\sframeIdx:\([-0-9]*\)\s.*/\1 \2 \3/g' > FlipTimes.log
echo FlipBoundTimes; sed -n '/LOG.*FlipLB.*FlipUB/p' $logfile | sed 's/sts:\([-0-9]*\)\s.*FrameIdx:\([-0-9]*\)Fliptime:\([-0-9]*\)\s.*FlipLB:\([-0-9]*\)\s.*FlipUB:\([-0-9]*\)\s.*Opto:\([-0-9]*\).*/\1 \2 \3 \4 \5 \6/g' > FlipBoundTimes.log # serverts, frame, lb, ub, state
echo StimEventTimes; sed -n '/STIMULUSEVENT/p' $logfile | sed 's/sts:\([-0-9]*\)\s.*ts:\([0-9]*\)\s.*{0,\([01]\)}.*/\1 \2 \3/g' > StimEventTimes.log
echo DataPacketTimes; sed -n '/DATAPACKET/p' $logfile | sed 's/sts:\([-0-9]*\)\s.*ts:\([-0-9]*\)\s.*/\1 \2/g' > DataPacketTimes.log
echo BufferSamples; sed -n '/BufferSamples/p' $logfile | sed 's/sts:\([-0-9]*\)\s.*ts:\([-0-9]*\)\s.*BufferSamples:\([-0-9]*\).*/\1 \2 \3/g' > BufferSamples.log
