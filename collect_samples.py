#!/usr/bin/env python

from __future__ import print_function, division
import sys, glob, gzip
import os.path


data_file = sys.argv[1]

# get the relative paths
dirname, basename = os.path.split(data_file)
filename, extension = os.path.splitext(basename)

# get the bundle samples and write them 
bundle_files = glob.glob('{0}/bundle-samples-{1}-cutoff*.csv.gz'.format(dirname, filename))
num_files = len(bundle_files)
bundle_collected_fp = gzip.open('{0}/bundle-samples-{1}.csv.gz'.format(dirname, filename), 'w')

for cutoff in xrange(1, num_files+1):
    fp = gzip.open('{0}/bundle-samples-{1}-cutoff{2}.csv.gz'.format(dirname, filename, cutoff))
    fp.readline()
    data_lines = fp.readlines()
    bundle_collected_fp.writelines(data_lines)
