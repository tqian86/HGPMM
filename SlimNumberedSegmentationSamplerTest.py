#!/usr/bin/env python
#-*-coding: utf-8 -*-

from __future__ import print_function

import sys, gzip, argparse, os.path
from SlimNumberedSegmentationSampler import *

def run():

    parser = argparse.ArgumentParser(description='Sampling program for the Numbered Segmentation Sampler.')
    parser.add_argument('--mode', required=True, choices=['batch', 'increm', 'predict'], help='specify the mode of the sampler, or as a predictor')
    parser.add_argument('--size', '-s', type=int, default=1000, help='sample size')
    parser.add_argument('--data', '-d', required=True, metavar='SUBJECT_FILE', help='gzipped individual subject RT data file (input)')
    parser.add_argument('--cutoff', type=int, help='use only up to the cutoff trial')
    parser.add_argument('--samplefile', required=True, metavar='SAMPLE_FILE', help='the file path of the resulting sample file (input or output)')
    parser.add_argument('--annealing', default='False', choices=['True', 'False', 'T', 'F'], help='turn on/off simulated annealing (temperature on prior)')
    parser.add_argument('--samplealpha', default='True', choices=['True', 'False', 'T', 'F'], help='turn on/off the sampling of alpha')
    parser.add_argument('--samplebeta', default='True', choices=['True', 'False', 'T', 'F'], help='turn on/off the sampling of beta')
    parser.add_argument('--usecontext', default='False', choices=['True', 'False', 'T', 'F'], help='turn on/off the use of context')
    parser.add_argument('--rsth', default=0.33, type=float, help='The threshold under which a localized gibbs will be run')
    parser.add_argument('--priortype', default='Geometric', choices=['Poisson', 'Geometric'], help='prior type on the length of a context')
    parser.add_argument('--alpha', default=1., type=float, help='alpha of CRP')
    parser.add_argument('--gps', default=0.1, type=float, help='gamma prior shape')
    parser.add_argument('--gpr', default=0.1, type=float, help='gamma prior rate')
    parser.add_argument('--gpa', default=1., type=float, help='geometric prior alpha')
    parser.add_argument('--gpb', default=1., type=float, help='geometric prior beta')
    parser.add_argument('--beta', default=1., type=float, help='beta of the DM prior')

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

    sampler = SlimNumberedSegmentationSampler(data_file = args.data, 
                                              sample_size = args.size, 
                                              s_type = args.mode,
                                              cutoff = args.cutoff,
                                              ialpha = args.alpha,
                                              ibeta = args.beta,
                                              annealing = args.annealing,
                                              sample_alpha = args.samplealpha,
                                              sample_beta = args.samplebeta,
                                              use_context = args.usecontext,
                                              resample_rate = args.rsth,
                                              prior_type = args.priortype,
                                              gamma_prior_shape = args.gps,
                                              gamma_prior_rate = args.gpr,
                                              geom_prior_alpha = args.gpa,
                                              geom_prior_beta = args.gpb,
                                              sample_output_file = sample_fp)
    # action!
    if args.mode == 'predict':
        sampler.load_samples()
        print('Loading finished.', file=sys.stderr)
    else:
        sampler.run()
        print('Sampling finished. Results are printed to the selected destination.', file=sys.stderr)
    
    print('Now printing predictions to stdout...', file=sys.stderr)
    #print('nseg.batch')
    #for trial in xrange(sampler.total_trial):
    #    if trial % 10 == 0: print(trial, file=sys.stderr)
    #    print(sampler.predict(int(trial)))

if __name__ == '__main__':
    run()
