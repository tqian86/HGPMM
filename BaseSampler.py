#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys, csv, gzip, random, copy, math, bisect
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

def uniqify(seq, idfun=None): 
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

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

    def __init__(self, data_file, sample_size=5000, cutoff=None, annealing=True, sample_output_file=sys.stdout):
        
        self.data = []
        self._import_data(data_file)
        self.original_data = copy.deepcopy(self.data)

        self.support = np.unique(self.data)
        self.support_size = len(self.support)
        if cutoff:
            self.data = self.data[:cutoff]
            self.context = self.context[:cutoff]
        self.iteration = 0
        self.sample_size = sample_size
        #self.data.insert(0,None)
        self.total_trial = len(self.data)
        self.sample_output_file = sample_output_file
        self.annealing = annealing in ['True', 'T', True]

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
            if self.iteration < self.total_trial * 0.2:
                self.temp = 0.2
            elif self.iteration < self.total_trial * 0.3:
                self.temp = 0.4
            elif self.iteration < self.total_trial * 0.4:
                self.temp = 0.6
            elif self.iteration < self.total_trial * 0.5:
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

        #int_labels = set(int_labels)
        #all_labels = set(xrange(1, max(int_labels) + 2))
        #return list(int_labels), min(all_labels - int_labels)        


if __name__ == '__main__':
    
    filename = sys.argv[1]
    bs = BaseSampler(filename)
    print(bs.support)
