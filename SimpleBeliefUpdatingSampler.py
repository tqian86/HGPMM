#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from BaseSampler import *
import sys, math, random
import numpy as np

def dgamma(x, shape, scale):
    return (x ** (shape - 1) * math.exp(-1 * x / scale)) / (math.gamma(shape) * scale ** shape)

class SimpleBeliefUpdatingSampler(BaseSampler):

    def __init__(self, data_file, sample_size, ialpha=1.0, s_type='batch', sample_output_file = sys.stdout):
        """Initialize the constructor.
        """
        BaseSampler.__init__(self, data_file, sample_size, cutoff=None, sample_output_file=sample_output_file)
        if s_type == 'batch': # if we are running a batch algorith, just initialize the first sample
            self.alpha = np.empty(self.sample_size+1, dtype="float32")
            self.alpha[0] = ialpha
        elif s_type == 'increm': # if we are running a particle filter, initialize all samples
            #self.alpha = np.linspace(0.001, 10, self.sample_size)
            self.thetas = np.random.dirichlet([ialpha] * self.support_size, self.sample_size)
            self.weight = np.ones(self.sample_size)
            self.weight /= np.sum(self.weight)
        self.s_type = s_type

    def copy_from_previous_iteration(self):
        """Copy the values of samples in the previous iteration 
        to the current iteration.
        """
        c_iter = self.iteration
        self.alpha[c_iter] = self.alpha[c_iter-1]
        
    def mh_sample_alpha(self, obs):
        """Sample the values of alpha given all the observations 
        up till Trial trial_t. If trial_t is None, use all observations.
        """
        c_iter = self.iteration
        proposal_sd = 2

        old_alpha = self.alpha[c_iter]
        new_alpha = np.random.gamma(shape = old_alpha, scale = proposal_sd)

        log_g_old, log_g_new = (0.0, 0.0) # flat prior
        
        # the first part
        log_g_old += math.lgamma(old_alpha * self.support_size) - \
            math.lgamma(old_alpha * self.support_size + len(obs))
        log_g_new += math.lgamma(new_alpha * self.support_size) - \
            math.lgamma(new_alpha * self.support_size + len(obs))

        # the second part
        for i in self.support:
            log_g_old += math.lgamma(obs.count(i) + old_alpha) - math.lgamma(old_alpha)
            log_g_new += math.lgamma(obs.count(i) + new_alpha) - math.lgamma(new_alpha)

        log_q_old = np.log(dgamma(old_alpha, shape = new_alpha, scale = proposal_sd))
        log_q_new = np.log(dgamma(new_alpha, shape = old_alpha, scale = proposal_sd))

        # compute the moving probability
        moving_prob = min(1, np.exp(log_g_new - log_g_old + log_q_old - log_q_new))
        
        u = random.uniform(0,1)
        if u < moving_prob: self.alpha[c_iter] = new_alpha

        return self.alpha[c_iter]        

    def pf_sample_alpha(self, obs):
        """Run the particle filter algorithm to obtain posterior samples of alpha
        incrementally.
        """
        for i in xrange(self.sample_size):
            c_alpha = self.alpha[i]
            log_p = 0
            log_p += math.lgamma(c_alpha * self.support_size) - math.lgamma(c_alpha * self.support_size + len(obs))
            for s in self.support:
                log_p += math.lgamma(obs.count(s) + c_alpha) - math.lgamma(c_alpha)
            self.weight[i] += log_p # temporarily store a log probability

        self.weight = self.weight - np.max(self.weight)

        # resample
        resampled_alpha = np.random.choice(self.alpha, size = self.sample_size, p = lognormalize(self.weight))
        #self.alpha = copy.deepcopy(resampled_alpha)
        return resampled_alpha

    def pf_sample_thetas(self, obs):
        """Run the particle filter algorithm to obtain posterior samples of thetas
        incrementally.
        """
        for i in xrange(self.sample_size):
            log_p = 0
            for s_i in xrange(self.support_size):
                log_p += obs.count(self.support[s_i]) * np.log(self.thetas[i][s_i])
            self.weight[i] += log_p
        
        self.weight = self.weight - np.max(self.weight)

        resampled_thetas_indice = np.random.choice(range(self.sample_size), size = self.sample_size, p = lognormalize(self.weight))
        resampled_thetas = self.thetas[resampled_thetas_indice]
        return resampled_thetas

    def run(self, end_trial = None, debug = True):
        """Run the sampler.
        """
        if self.s_type == 'batch':
            # headers
            headers = 'alpha'
            if debug: print(headers, file=self.sample_output_file)

            if end_trial:
                obs = [trial['pos'] for trial in self.data[:end_trial+1]]
            else:
                obs = [trial['pos'] for trial in self.data]

            for i in xrange(self.sample_size):
                self.iteration += 1
                self.copy_from_previous_iteration()
                self.mh_sample_alpha(obs)

                if debug: self.print_current_iteration(dest=self.sample_output_file)
        
        elif self.s_type == 'increm':
            headers = ['iter', 'pos']
            for s in self.support:
                headers.append('theta_' + str(s))
            print(*headers, sep=',', file=self.sample_output_file)

            initial_thetas = np.round(self.thetas, decimals=3)
            for particle in initial_thetas:
                print(0, None, *particle, sep=',', file=self.sample_output_file)
            
            for i in xrange(self.total_trial):
                if i % 100 == 0: print('Processed {0} trials...'.format(i), file=sys.stderr)
                obs = [trial['pos'] for trial in self.data[:i+1]]
                #obs = [trial['pos'] for trial in self.data[:i+1][-10:]]
                #resampled_alpha = np.round(self.pf_sample_alpha(obs), decimals=3)
                #if debug: print(*resampled_alpha, sep=',', file=self.sample_output_file)
                resampled_thetas = np.round(self.pf_sample_thetas(obs), decimals=3)
                for particle in resampled_thetas:
                    print(i+1, self.data[i]['pos'], *particle, sep=',', file=self.sample_output_file)
                
        self.sample_output_file.close()

    def print_current_iteration(self, dest):
        """Print samples to the specified destination.
        """
        output = ''
        # display alpha information
        output += str(self.alpha[self.iteration])

        print(output, file=dest)

    def load_samples(self):
        """If the sampler is run in the `predict' mode,
        load the samples from sample file.
        """
        i = 0
        reader = csv.reader(self.sample_output_file)
        for row in reader:
            if row[0] == 'beta': continue
            self.beta[i] = row[0]
            self.l[i] = row[1]
            self.breakpoints[i] = row[2:]
            i += 1

    def predict(self):
        
        return
