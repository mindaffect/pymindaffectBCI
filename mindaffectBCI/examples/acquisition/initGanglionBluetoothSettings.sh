#!/bin/bash
intervalmin=6
intervalmax=9
latency=0
if [ $# -gt 0 ] ; then
  intervalmin=$1
  intervalmax=$1
  if [ $# -gt 1 ] ; then
    intervalmax=$2
  fi
fi
echo "$intervalmin" > /sys/kernel/debug/bluetooth/hci0/conn_min_interval
echo conn_min_interval; cat /sys/kernel/debug/bluetooth/hci0/conn_min_interval

echo "$intervalmax" > /sys/kernel/debug/bluetooth/hci0/conn_max_interval
echo conn_max_interval; cat /sys/kernel/debug/bluetooth/hci0/conn_max_interval

echo "$latency" > /sys/kernel/debug/bluetooth/hci0/conn_latency
