#!/usr/bin/env python

from __future__ import print_function, division
import sys, glob, gzip
import os.path

data_file = sys.argv[1]

num_files = len(gzip.open(data_file).readlines()) - 1

# get the relative paths
dirname, basename = os.path.split(data_file)
filename = basename.split('.')[0]

# get the bundle samples and write them 
print("Processing bundle samples")
bundle_files = glob.glob('{0}/bundle-samples-{1}-cutoff*.csv.gz'.format(dirname, filename))
bundle_collected_fp = gzip.open('{0}/bundle-samples-{1}.csv.gz'.format(dirname, filename), 'w')

bundle_header = "cutoff,iteration,loglik,alpha,l," + ",".join([str(_) for _ in range(1, num_files + 1)]) + '\n'

bundle_collected_fp.write(bundle_header)

for cutoff in xrange(1, num_files+1):
    fp = gzip.open('{0}/bundle-samples-{1}-cutoff{2}.csv.gz'.format(dirname, filename, cutoff))
    fp.readline()
    data_lines = fp.readlines()
    data_lines = [str(cutoff) + ',' + _ for _ in data_lines]
    bundle_collected_fp.writelines(data_lines)

bundle_collected_fp.close()

# get the beta samples and write them 
print("Processing beta samples")
beta_files = glob.glob('{0}/beta-samples-{1}-cutoff*.csv.gz'.format(dirname, filename))
beta_collected_fp = gzip.open('{0}/beta-samples-{1}.csv.gz'.format(dirname, filename), 'w')

beta_header = "cutoff,iteration,category,beta\n"

beta_collected_fp.write(beta_header)

for cutoff in xrange(1, num_files+1):
    fp = gzip.open('{0}/beta-samples-{1}-cutoff{2}.csv.gz'.format(dirname, filename, cutoff))
    fp.readline()
    data_lines = fp.readlines()
    beta_collected_fp.writelines(data_lines)

beta_collected_fp.close()

# get the predictions samples and write them 
print("Processing predictions")
predictions_files = glob.glob('{0}/predictions-{1}-cutoff*.csv.gz'.format(dirname, filename))
predictions_collected_fp = gzip.open('{0}/predictions-{1}.csv.gz'.format(dirname, filename), 'w')

predictions_header = "t,pos,probability\n"

predictions_collected_fp.write(predictions_header)

for cutoff in xrange(1, num_files+1):
    fp = gzip.open('{0}/predictions-{1}-cutoff{2}.csv.gz'.format(dirname, filename, cutoff))
    fp.readline()
    data_lines = fp.readlines()    
    predictions_collected_fp.writelines(data_lines)

predictions_collected_fp.close()
