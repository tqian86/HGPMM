#!/bin/sh

lines=`zcat < $1 | wc -l`
echo $lines
for i in `seq 1 $lines`
do
    ./SlimNumberedSegmentationSamplerTest.py --data $1 --cutoff $i --search
done
