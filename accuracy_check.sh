#!/bin/bash

if [[ ! -e "maskrcnn_cpu.log" ]]; then
    echo "The maskrcnn cpu inference log doesn\'t exist."
    exit -1
fi

if [[ ! -e "maskrcnn_gpu.log" ]]; then
    echo "The maskrcnn gpu inference log doesn\'t exist."
    exit -1
fi

ap_cpu=`grep "Average " maskrcnn_cpu.log | awk -F '=' '{print $5}'`
ap_gpu=`grep "Average " maskrcnn_gpu.log | awk -F '=' '{print $5}'`

echo Now check the average percision/recall value to see if the cpu pr is larger than or equal to cpu pr.

x=12
for ((i=1;i<=x;i++)); do
    cpu=`echo $ap_cpu | awk -v id=$i -F ' ' '{print $id}'`;
    gpu=`echo $ap_gpu | awk -v id=$i -F ' ' '{print $id}'`;
    RESULT=`echo "$cpu >= $gpu" | bc`;
    if [ $RESULT -eq 1 ]; then
	echo "Test Check(tolerance number): Pass"
	exit 0
    else
	echo "Test Check(tolerance number): Fail"
	exit 1
    fi
done
