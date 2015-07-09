#!/bin/sh

lines=`zcat $1 | wc -l`
echo $lines
