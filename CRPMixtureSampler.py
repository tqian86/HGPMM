#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from BaseSampler import *
import sys, math, random, copy
import numpy as np

class CRPMixtureSampler(BaseSampler):

    def __init__(self, data_file, sample_size, ibeta=.001, ialpha=.5, s_type='batch', sample_output_file = sys.stdout):
        """Initialize the constructor.
        """
        BaseSampler.__init__(self, data_file, sample_size)
        self.alpha = ialpha
        self.context = np.empty((self.sample_size + 1, self.total_trial), dtype='int16')
        self.beta = np.empty(self.sample_size + 1)

        # initialize the samples
        self.context[0] = [random.randint(1,1) for _ in xrange(self.total_trial)]
        self.beta[0] = ibeta

        # initialize pointer
        self.iter = 1
        self.s_type = s_type

        self.sample_output_file = sample_output_file

    def copy_previous_sample(self):
        """Copy the values of samples in the previous iteration
        to the current iteration.
        """
        self.context[self.iter] = self.context[self.iter-1]
        self.beta[self.iter] = self.beta[self.iter-1]

    def batch_sample_beta(self):
        """Perform Metropolis Hastings sampling on beta.
        """
        c_contexts = self.context[self.iter]

        old_beta = self.beta[self.iter]
        new_beta = -1
        proposal_sd = .1
        while new_beta <= 0:
            new_beta = random.gauss(mu = old_beta, sigma = proposal_sd)
        
        # set up to calculate the g densities for both the old and new beta values
        log_g_old = -1 * old_beta # which is np.log(np.exp(-1 * old_beta))
        log_g_new = -1 * new_beta # similar as above

        # derive contexts from breakpoints arrangement
        context_dict = self.make_context_dict(c_contexts)
        for context in context_dict.keys():
            log_g_old += math.lgamma(self.support_size * old_beta) \
                - math.lgamma(self.support_size * old_beta + len(context_dict[context]))
            log_g_new += math.lgamma(self.support_size * new_beta) \
                - math.lgamma(self.support_size * new_beta + len(context_dict[context]))
            
            for y in self.support:
                log_g_old += math.lgamma(context_dict[context].count(y) + old_beta) - math.lgamma(old_beta)
                log_g_new += math.lgamma(context_dict[context].count(y) + new_beta) - math.lgamma(new_beta)

        # compute candidate densities q for old and new beta
        # since the proposal distribution is normal this step is not needed
        log_q_old = 0#np.log(dnorm(old_beta, loc = new_beta, scale = proposal_sd))
        log_q_new = 0#np.log(dnorm(new_beta, loc = old_beta, scale = proposal_sd)) 
        
        # compute the moving probability
        moving_prob = min(1, np.exp((log_g_new + log_q_old) - (log_g_old + log_q_new)))
        
        u = random.uniform(0,1)
        if u < moving_prob: self.beta[self.iter] = new_beta
        return self.beta[self.iter]

    def batch_sample_context(self):
        """Assuming all trials are observed at once, sample
        the context variables of each trial.
        """
        c_beta = self.beta[self.iter]
        
        # sample the context of each trial
        for i in xrange(self.total_trial):
            c_contexts = self.context[self.iter]
            c_pos = self.data[i]['pos']
            context_dict = self.make_context_dict(c_contexts, excluded = i)
            context_grid = context_dict.keys()
            context_grid.append(self.smallest_unused_label(context_grid))
            context_p_grid = np.empty(len(context_grid))

            for context in context_grid:
                try: 
                    context_size = len(context_dict[context])
                    prior = context_size / (self.total_trial + self.alpha)
                    likelihood = (context_dict[context].count(c_pos) + c_beta) \
                        / (context_size + self.support_size * c_beta)
                except KeyError:
                    prior = self.alpha / (self.total_trial + self.alpha)
                    likelihood = 1.0 / self.support_size
                
                context_p_grid[context_grid.index(context)] = prior * likelihood
            
            context_p_grid /= sum(context_p_grid)
            #print('pos:', c_pos)
            #print(context_grid)
            #print(context_p_grid)
            #raw_input()
            self.context[self.iter, i] = sample(context_grid, context_p_grid)

        return True

    def remap_context_labels(self):
        """Reorder context labels so it's always ascending.
        """
        c_contexts = list(self.context[self.iter])
        unique_contexts = uniqify(c_contexts)
        remap_dict = dict(zip(unique_contexts,
                              range(1, len(unique_contexts) + 1)))

        remapped = copy.deepcopy(self.context[self.iter])
        for old, new in remap_dict.iteritems():
            self.context[self.iter][remapped==old] = new

    def run(self):
        """Run the sampler.
        """
        self.iter = 0
        self.remap_context_labels()
        for i in xrange(self.sample_size):
            self.iter = i + 1
            self.copy_previous_sample()
            if self.s_type == 'batch':
                self.batch_sample_context()
                self.batch_sample_beta()
            self.remap_context_labels()
            self.print_current_iteration(dest = self.sample_output_file)
            if i % 500 == 0: self.sample_output_file.flush()

        self.sample_output_file.close()

    def print_current_iteration(self, dest):
        """Print out the current sample
        """
        header = ''
        header += 'beta,'
        header += ','.join([str(t) for t in xrange(1, self.total_trial+1)])
        if self.iter == 1: print(header, file = dest)

        output = ''
        # display beta information
        output += str(self.beta[self.iter]) + ','
        # display category information
        output += ','.join([str(c) for c in self.context[self.iter]])
        print(output, file = dest)

    def make_context_dict(self, contexts, excluded=None):
        context_dict = {}
        for i in xrange(len(contexts)):
            if excluded is not None and i == excluded: continue
            context = contexts[i]
            try: context_dict[context].append(self.data[i]['pos'])
            except KeyError: context_dict[context] = [self.data[i]['pos']]
        return context_dict

    def smallest_unused_label(self, int_labels):
        
        if len(int_labels) == 0: return 1
        int_labels = set([int(i) for i in int_labels])
        all_labels = set(xrange(1, max(int_labels) + 2))
        return min(all_labels - int_labels)        

