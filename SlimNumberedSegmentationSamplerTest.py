#!/usr/bin/env python
#-*-coding: utf-8 -*-

from __future__ import print_function

import sys, gzip, argparse, os.path
from SlimNumberedSegmentationSampler import *

def run():

    parser = argparse.ArgumentParser(description='Sampling program for the Numbered Segmentation Sampler.')
    parser.add_argument('--size', '-s', type=int, default=1000, help='sample size')
    parser.add_argument('--data', '-d', required=True, metavar='SUBJECT_FILE', help='gzipped individual subject RT data file (input)')
    parser.add_argument('--cutoff', '-c', type=int, help='use only up to the cutoff trial')
    parser.add_argument('--output_to_stdout', action='store_true', help='in addition to saving samples to files, also print to screen')
    parser.add_argument('--annealing', default='False', choices=['True', 'False', 'T', 'F'], help='turn on/off simulated annealing (temperature on prior)')
    parser.add_argument('--samplealpha', default='False', choices=['True', 'False', 'T', 'F'], help='turn on/off the sampling of alpha')
    parser.add_argument('--samplebeta', default='True', choices=['True', 'False', 'T', 'F'], help='turn on/off the sampling of beta')
    parser.add_argument('--usecontext', default='False', choices=['True', 'False', 'T', 'F'], help='turn on/off the use of context')
    parser.add_argument('--search', action='store_true', help='use stochastic search instead')
    parser.add_argument('--mumble', action='store_true', help='print debug information to standard error')
    parser.add_argument('--priortype', default='Poisson', choices=['Poisson', 'Geometric'], help='prior type on the length of a context')
    parser.add_argument('--alpha', default=1., type=float, help='alpha of CRP')
    parser.add_argument('--gps', default=0.1, type=float, help='gamma prior shape')
    parser.add_argument('--gpr', default=0.1, type=float, help='gamma prior rate')
    parser.add_argument('--gpa', default=1., type=float, help='geometric prior alpha')
    parser.add_argument('--gpb', default=1., type=float, help='geometric prior beta')
    parser.add_argument('--beta', default=1., type=float, help='beta of the DM prior')

    args = parser.parse_args()

    sampler = SlimNumberedSegmentationSampler(data_file = args.data, 
                                              sample_size = args.size, 
                                              cutoff = args.cutoff,
                                              ialpha = args.alpha,
                                              ibeta = args.beta,
                                              annealing = args.annealing,
                                              sample_alpha = args.samplealpha,
                                              sample_beta = args.samplebeta,
                                              use_context = args.usecontext,
                                              prior_type = args.priortype,
                                              poisson_prior_shape = args.gps,
                                              poisson_prior_rate = args.gpr,
                                              geom_prior_alpha = args.gpa,
                                              geom_prior_beta = args.gpb,
                                              output_to_stdout = args.output_to_stdout,
                                              record_best = args.search,
                                              debug_mumble = args.mumble)
    # action!
    print('Figuring out bundles in %s (cutoff = %s) ...' % (args.data, args.cutoff))
    sampler.run()
    print('Done. Results are saved in the same directory as the input file.\n', file=sys.stderr)
    
if __name__ == '__main__':
    run()
