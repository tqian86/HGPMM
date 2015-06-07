#!/usr/bin/env python
#-*-coding: utf-8 -*-

from __future__ import print_function

import sys, gzip, argparse, os.path
from SimpleBeliefUpdatingSampler import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sampling program for the Simple Belief-updating Sampler.')
    parser.add_argument('--mode', required=True, choices=['batch', 'increm', 'predict'], help='specify the mode of the sampler, or as a predictor')
    parser.add_argument('--size', '-s', type=int, default=1000, help='sample size')
    parser.add_argument('--data', '-d', required=True, help='gzipped individual subject RT data file (input)')
    parser.add_argument('--samplefile', required=True, help='the file path of the resulting sample file (input or output)')
    parser.add_argument('--thinning', type=int, default=1, help='the thinning parameter of the gibbs sampler')
    parser.add_argument('--burnin', type=int, default=500, help='number of burn-in samples')
    
    args = parser.parse_args()

    # check for inconsistencies and potential errors
    if args.mode == 'predict':
        if not os.path.exists(args.samplefile):
            print('Sorry, but the specified sample file', args.samplefile, 'does not exist!', file=sys.stderr)
            sys.exit(0)
    else:
        if os.path.exists(args.samplefile):
            print('Sorry, but the specified sample file', 
                  args.samplefile, 
                  'already exists! Please manually delete the file before running the sampler.',
                  file=sys.stderr)
            sys.exit(0)

    # make sure the sample file is opened in the correct manner
    # depending on which mode the program is run in
    if args.mode == 'predict':
        sample_fp = gzip.open(args.samplefile, 'r')
    else:
        sample_fp = gzip.open(args.samplefile, 'w')

    sampler = SimpleBeliefUpdatingSampler(data_file = args.data, 
                                          sample_size = args.size, 
                                          s_type = args.mode,
                                          sample_output_file = sample_fp)
    # action!
    if args.mode == 'predict':
        sampler.load_samples()
        print('Loading finished.', file=sys.stderr)
    else:
        sampler.run()
        print('Sampling finished. Results printed to the selected destination.', file=sys.stderr)
    
    #print('Now printing predictions to stdout...', file=sys.stderr)
    #print('seg.batch')
    #for trial in xrange(sampler.total_trial):
        #if trial % 100 == 0: print('Trial:', trial, file=sys.stderr)
        #print(sampler.predict(trial, burnin = args.burnin, thinning = args.thinning))
