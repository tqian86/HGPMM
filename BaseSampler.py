#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import sys, csv, gzip, random, copy, math, bisect, os.path
import numpy as np

def printerr(x):
    print(*x, file=sys.stderr)

def lognormalize(x):
    # adapt it to numpypy
    x = x - np.max(x)
    a = np.log(np.sum([np.exp(e) for e in x]))
    return np.exp(x - a)

def sample(samples, prob):
    """Step sample from a discrete distribution using CDF
    """
    n = len(samples)
    r = random.random() # range: [0,1)
    total = 0           # range: [0,1]
    for i in xrange(n):
        total += prob[i]
        if total > r:
            return samples[i]
    raise Exception('distribution not normalized')

def thin_list(lst):
    thin_lst = []
    thin_lst.append(lst[0])
    for i in xrange(1, len(lst)):
        if thin_lst[-1] != lst[i]: thin_lst.append(lst[i])
    return thin_lst

def cdfpois(y, rate):
    """Calculate the cumulative probability of a poisson dist
    up to y.
    """
    p = 0
    for i in xrange(1, y+1):
        p += math.exp(log_dpois(y = i, rate = rate))
    return p

def dgamma(x, shape, scale):
    return (x ** (shape - 1) * math.exp(-1 * x / scale)) / (math.gamma(shape) * scale ** shape)

def log_dpois(y, rate):
    return -rate + y * math.log(rate) - math.log(math.factorial(y))

def log_dnbinom(y, alpha, beta):
    return math.lgamma(alpha+y) - math.lgamma(y+1) - math.lgamma(alpha) + \
        alpha * (math.log(beta) - math.log(beta+1)) + \
        y * (0 - math.log(beta+1))

class BaseSampler:

    def __init__(self, data_file, sample_size=5000, cutoff=None, annealing=False, output_to_stdout = False, record_best=True):
        
        self.data = []
        self._import_data(data_file)
        self.original_data = copy.deepcopy(self.data)

        self.support = np.unique(self.data)
        self.support_size = len(self.support)
        if cutoff:
            self.cutoff = cutoff
            self.data = self.data[:cutoff]
            self.context = self.context[:cutoff]
        self.iteration = 0
        self.sample_size = sample_size
        self.N = len(self.data)
        self.output_to_stdout = output_to_stdout
        self.annealing = annealing in ['True', 'T', True]
        self.best_sample = (None, None) # (sample, loglikelihood)
        self.record_best = record_best
        self.best_diff = []
        self.no_improv = 0

    def _import_data(self, data_file):

        with gzip.open(data_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.data.append(row)
        
        self.context = [''.join([self._get_context(_, 'n.hipo'),
                                 self._get_context(_, 'n.rabbit'),
                                 self._get_context(_, 'n.snail'),
                                 self._get_context(_, 'n.dinasour')])
                        for _ in self.data]
        self.data = [_['pos'] for _ in self.data]

        self.source_filepath = data_file
        self.source_dirname = os.path.dirname(data_file) + '/'
        self.source_filename = os.path.basename(data_file).split('.')[0]
        
        
    def _get_context(self, row, feature_name):
        if feature_name in row:
            return row[feature_name]
        else:
            return ''
        
    def set_temperature(self):
        """Set the temperature of simulated annealing
        as a function of sampling progress.
        """
        if self.annealing is False: 
            self.temp = 1.0
            return

        if self.s_type == 'batch':
            if self.iteration < self.sample_size * 0.2:
                self.temp = 0.2
            elif self.iteration < self.sample_size * 0.3:
                self.temp = 0.4
            elif self.iteration < self.sample_size * 0.4:
                self.temp = 0.6
            elif self.iteration < self.sample_size * 0.5:
                self.temp = 0.8
            else:
                self.temp = 1.0

        elif self.s_type == 'increm':
            if self.iteration < self.N * 0.2:
                self.temp = 0.2
            elif self.iteration < self.N * 0.3:
                self.temp = 0.4
            elif self.iteration < self.N * 0.4:
                self.temp = 0.6
            elif self.iteration < self.N * 0.5:
                self.temp = 0.8
            else:
                self.temp = 1.0

    def smallest_unused_label(self, int_labels):
        
        if len(int_labels) == 0: return [], [], 1
        label_count = np.bincount(int_labels)
        try: new_label = np.where(label_count == 0)[0][1]
        except IndexError: new_label = max(int_labels) + 1
        uniq_labels = np.unique(int_labels)
        return label_count, uniq_labels, new_label

    def auto_save_sample(self, sample):
        """Save the given sample as the best sample if it yields
        a larger log-likelihood of data than the current best.
        """
        new_logprob = self._logprob(sample)
        # if there's no best sample recorded yet
        if self.best_sample[0] is None and self.best_sample[1] is None:
            self.best_sample = (sample, new_logprob)
            print('Initial sample generated, loglik: {0}'.format(new_logprob), file=sys.stderr)
            return

        # if there's a best sample
        if new_logprob > self.best_sample[1]:
            self.no_improv = 0
            self.best_diff.append(new_logprob - self.best_sample[1])
            self.best_sample = (copy.deepcopy(sample), new_logprob)
            print('New best sample found, loglik: {0}'.format(new_logprob), file=sys.stderr)
            return True
        else:
            self.no_improv += 1
            return False

    def no_improvement(self, threshold=500):
        if len(self.best_diff) == 0: return False
        if self.no_improv > threshold or np.mean(self.best_diff[-threshold:]) < .1:
            print('Too little improvement in loglikelihood for %s iterations - Abort searching' % threshold, file=sys.stderr)
            return True
        return False

    def _logprob(self, sample):
        """Compute the logliklihood of data given a sample. This method
        does nothing in the base class.
        """
        return 0

    
if __name__ == '__main__':
    
    filename = sys.argv[1]
    bs = BaseSampler(filename)
    print(bs.support)
