#!/usr/bin/env python
#-*-coding: utf-8 -*-

from __future__ import print_function

import sys, gzip, pprint
from DPMixtureSeqSampler import *


if __name__ == '__main__':

    arg = sys.argv
    subject_file = arg[1]
    ss = int(arg[2])

    dppf = DPMixtureSeqSampler(data_file = subject_file, sample_size = ss)
    total_trial = len(dppf.data) - 1
    print('target.mmwv')
    #dppf.print_sampler_config()
    for i in xrange(total_trial):
        print(dppf.predict(verbose=False))
        dppf.filter()
        #if i % 50 == 0:
        #    unique_contexts = [set(p[0]) for p in dppf.particles]
        #    pprint.pprint(unique_contexts, stream=sys.stderr)
        if i == total_trial-1:
            print(dppf.particles[0][0], file=sys.stderr)
