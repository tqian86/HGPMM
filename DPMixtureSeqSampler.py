#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from BaseSampler import *
import sys, math, random
import numpy as np

class DPMixtureSeqSampler(BaseSampler):

    def __init__(self, data_file, sample_size, ibeta=1.0, ialpha=1.0):
        """Initialize the constructor.
        """
        BaseSampler.__init__(self, data_file, sample_size)
        self.alpha = ialpha
        contexts = [[]] * self.sample_size
        betas = [ibeta] * self.sample_size
        self.particles = zip(contexts, betas)
        self.weights = np.ones(sample_size) / sample_size
        self.cursor_index = 0

    def print_sampler_config(self):
        """Print the current configurations of the sampler.
        """
        print('Number of particles:', self.sample_size)
        print('Particles:', self.particles[0], '...', self.particles[self.sample_size - 1])
        print('Weights:', self.weights[0], '...', self.weights[self.sample_size-1])
        print('Number of observed instances:', self.cursor_index)

    def predict(self, verbose=False):
        """Predict the probability distribution over holes,
        marginalized over contexts.
        """
        prob = 0
        next_hole = self.data[self.cursor_index]['pos']
        for p in self.particles:
            context_dict = self.make_context_dict(p[0])
            beta = p[1]
            p_next_hole = 0
            candidate_contexts = context_dict.keys()
            candidate_contexts.append(self.smallest_unused_label(candidate_contexts))

            for context in candidate_contexts:
                prior = self.get_prior_prob(context_dict, context)
                likelihood = self.get_likelihood(context_dict, context, next_hole, beta)
                p_next_hole += prior * likelihood
            
            prob += p_next_hole
        
        if verbose:
            print('The next appearance is at hole:', next_hole)
            print('Predicted probability:', prob / self.sample_size)
        return prob / self.sample_size

    def filter(self, verbose=False):
        """Update all particles to reflect the latest observation.
        """
        self.cursor_index += 1
        new_particles = []
        new_weights = []
        eta = 0
        for i in xrange(self.sample_size):
            # sample an existing particle according to current weights
            old_contexts, old_beta = sample(self.particles, self.weights)
            old_context_dict = self.make_context_dict(old_contexts)

            # generate a new beta
            new_beta = old_beta + random.gauss(mu = 0, sigma = 0.01)
            while new_beta < 0:
                new_beta = old_beta + random.gauss(mu = 0, sigma = 0.01)
            # generate new contexts
            new_contexts = old_contexts + [self.sample_from_crp(old_context_dict)]
            # weight the new particle
            new_weight = self.get_likelihood(context_dict = old_context_dict,
                                             context = new_contexts[-1],
                                             hole = self.data[self.cursor_index]['pos'],
                                             beta = new_beta)
            eta += new_weight
            new_particles.append((new_contexts, new_beta))
            new_weights.append(new_weight)
        
        # normalize the weights
        self.particles = new_particles
        self.weights = np.array(new_weights) / eta
        if verbose: print(self.particles, file=sys.stderr)

    def smallest_unused_label(self, int_labels):
        
        if len(int_labels) == 0: return 1
        int_labels = set([int(i) for i in int_labels])
        all_labels = set(xrange(1, max(int_labels) + 2))
        return min(all_labels - int_labels)        

    def make_context_dict(self, contexts):
        context_dict = {}
        for i in xrange(len(contexts)):
            context = contexts[i]
            try: context_dict[context].append(self.data[i+1]['pos'])
            except KeyError: context_dict[context] = [self.data[i+1]['pos']]
        return context_dict

    def get_prior_prob(self, context_dict, context):
        
        if len(context_dict) == 0:
            N = 0
        else:
            N = len(reduce(list.__add__, context_dict.values()))
        if context in context_dict:
            return len(context_dict[context]) / (self.alpha + N)
        else:
            return self.alpha / (self.alpha + N)
        
    def get_likelihood(self, context_dict, context, hole, beta):
        
        if context in context_dict:
            return (context_dict[context].count(hole) + beta) / (len(context_dict[context]) + 4 * beta)
        else:
            return beta / (4 * beta)        
    
    def sample_from_crp(self, context_dict):
        """Sample from a Chinese Restaurant Process. i.e.
        sample a table label given existing ones.
        """
        candidate_contexts = (context_dict.keys())
        candidate_contexts.append(self.smallest_unused_label(candidate_contexts))
        prob = []
        for context in candidate_contexts:
            prob.append(self.get_prior_prob(context_dict, context))
        return sample(candidate_contexts, prob)

    def mh_sample_alpha(self, obs):
        """Sample the values of alpha given all the observations 
        up till Trial trial_t. If trial_t is None, use all observations.
        """
        c_iter = self.iteration
        proposal_sd = .1

        old_alpha = self.alpha[c_iter]
        new_alpha = 0
        while new_alpha <= 0:
            new_alpha = random.gauss(mu = old_alpha, sigma = proposal_sd)
        
        log_g_old, log_g_new = (0.0, 0.0) # flat prior
        
        # the first part
        log_g_old += math.lgamma(old_alpha * 4) - math.lgamma(old_alpha * 4 + len(obs))
        log_g_new += math.lgamma(new_alpha * 4) - math.lgamma(new_alpha * 4 + len(obs))

        # the second part
        for i in xrange(4):
            log_g_old += math.lgamma(obs.count(i) + old_alpha) - math.lgamma(old_alpha)
            log_g_new += math.lgamma(obs.count(i) + new_alpha) - math.lgamma(new_alpha)

        # compute the moving probability
        moving_prob = min(1, np.exp(log_g_new - log_g_old))
        
        u = random.uniform(0,1)
        if u < moving_prob: self.alpha[c_iter] = new_alpha

        return self.alpha[c_iter]
