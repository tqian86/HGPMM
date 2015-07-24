#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
cimport cython
from clsampler import BaseSampler, sample, lognormalize
import numpy as np; cimport numpy as np
from scipy.stats import poisson, gamma
from scipy.stats import beta as beta_dist
from time import time
from collections import Counter
import bisect, gzip, random, math, sys

from libc.math cimport log, lgamma

cdef double log_dpois(double y, double rate):
    return -rate + y * log(rate) - lgamma(y+1)

cdef double log_dgamma(double x, double shape, double scale):
    return (shape - 1) * log(x) + (-1 * x / scale) - lgamma(shape) - shape * log(scale)

def smallest_unused_label(int_labels):
    
    if len(int_labels) == 0: return np.array([]), np.array([]), 1
    label_count = np.bincount(int_labels)
    try: new_label = np.where(label_count == 0)[0][1]
    except IndexError: new_label = max(int_labels) + 1
    uniq_labels = np.unique(int_labels)
    return label_count, uniq_labels, new_label

class SlimNumberedSegmentationSampler(BaseSampler):

    def __init__(self, sample_size = 1000, cl_mode = False, double ialpha = 1, double ibeta=1, cutoff=None, annealing=False,
                 output_to_stdout = False, record_best = False, debug_mumble = False,
                 sample_alpha = True, sample_beta = True, use_context = False,
                 prior_type = 'Poisson', double poisson_prior_shape = 1, double poisson_prior_rate = 1,
                 double geom_prior_alpha = 1, double geom_prior_beta = 1):
        """Initialize the sampler.
        """
        BaseSampler.__init__(self,
                             cl_mode = cl_mode,
                             sample_size = sample_size,
                             cutoff = cutoff,
                             annealing = annealing,
                             output_to_stdout = output_to_stdout,
                             record_best = record_best,
                             debug_mumble = debug_mumble)
        # other shared parameters
        self.prior_type = prior_type
        self.sample_alpha = sample_alpha in ['T', 'True', True]
        self.sample_beta = sample_beta in ['T', 'True', True]
        self.use_context = use_context in ['T', 'True', True]
        self.ibeta = ibeta

        if self.prior_type == 'Poisson':
            self.poisson_prior_shape = poisson_prior_shape
            self.poisson_prior_rate = poisson_prior_rate
        elif self.prior_type == 'Geometric':
            self.geom_prior_alpha = geom_prior_alpha
            self.geom_prior_beta = geom_prior_beta

        cdef list bundles, categories
        bundles = [0]
        self.bundles = bundles
        categories = [1]
        self.categories = categories

        self.alpha = ialpha
        
        cdef dict beta
        beta = {1:ibeta}
        self.beta = beta

        cdef double l
        if self.prior_type == 'Poisson':
            l = np.random.gamma(shape = self.poisson_prior_shape, scale = 1. / self.poisson_prior_rate)
        elif self.prior_type == 'Geometric':
            l = np.random.beta(a = self.geom_prior_alpha, b = self.geom_prior_beta)
        self.l = l

    def read_csv(self, filepath, header=True):
        if self.use_context:
            BaseSampler.read_csv(self,
                                 filepath = filepath,
                                 obs_vars = ['pos', 'n.hipo', 'n.rabbit', 'n.snail', 'n.dinasour'],
                                 header = header)
        else:
            BaseSampler.read_csv(self, filepath = filepath, obs_vars = ['pos'], header = header)
            self.data = np.ravel(self.data, order='C')
            self.support = np.unique(self.original_data)
            self.support_size = len(self.support)

    def batch_sample_bundles(self):
        """Perform Gibbs sampling on clusters.
        """
        cdef int nth, left_run_cat, original_idx, would_be_idx, outcome, next_run_cat, c
        cdef double left_run_beta, right_run_beta, together_beta
        cdef list left_run, right_run
        cdef list grid
        cdef dict cat_dict, cat_count_dict
        cdef np.ndarray[np.float_t, ndim=1] log_p_grid
        
        # we index the sequence of observations using "nth"
        _, _, new_cat = smallest_unused_label(self.categories)
        for nth in xrange(1, self.N):
            a_time = time()
            if nth in self.bundles:
                
                original_idx = self.bundles.index(nth)
                cat_dict = self.get_category_flat_dict(avoid = [original_idx, original_idx - 1])

                # get the left run
                left_run_cat = self.categories[original_idx-1]
                left_run_beta = self.beta[left_run_cat]
                left_run = list(self.data[self.bundles[original_idx-1]:self.bundles[original_idx]])

                # get the right run
                right_run_cat = self.categories[original_idx]
                right_run_beta = self.beta[right_run_cat]
                try: right_run = list(self.data[self.bundles[original_idx]:self.bundles[original_idx+1]])
                except: right_run = list(self.data[self.bundles[original_idx]:])
                
                together_beta = left_run_beta

                # remove
                del self.bundles[original_idx]
                del self.categories[original_idx]
                if self.categories.count(right_run_cat) == 0:
                    del self.beta[right_run_cat]
            else:
                would_be_idx = bisect.bisect(self.bundles, nth)
                cat_dict = self.get_category_flat_dict(avoid = [would_be_idx - 1])

                left_run_cat = self.categories[would_be_idx-1]
                left_run_beta = self.beta[left_run_cat]

                right_run_cat = None
                right_run_beta = left_run_beta

                together_beta = left_run_beta

                # get the left and right runs of a breakpoint
                left_run, right_run = self.get_surround_runs(bps = self.bundles, target_bp = nth)
                
            # set up the grid
            grid = [0, 1]
            log_p_grid = np.empty(len(grid))

            # pre-count
            cat_count_dict = {}
            for c in cat_dict.keys():
                cat_count_dict[c] = Counter(cat_dict[c])

            # compute the prior probability of each run
            log_p_grid[0] = self.log_length_prior(runs = [left_run + right_run]).sum() 
            log_p_grid[1] = self.log_length_prior(runs = [left_run, right_run]).sum() 
            
            # compute the likelihood of each case
            try:
                if original_idx == len(self.bundles) - 1: next_run_cat = None
                else: next_run_cat = self.categories[original_idx + 1]
                log_p_grid[0] += self.log_cond_prob(obs = left_run + right_run, cat = left_run_cat,
                                                    cat_dict = cat_dict, cat_count_dict = cat_count_dict,
                                                    beta = together_beta, avoid_cat = next_run_cat)
            except:
                log_p_grid[0] += self.log_cond_prob(obs = left_run + right_run, cat = left_run_cat,
                                                    cat_dict = cat_dict, cat_count_dict = cat_count_dict, beta = together_beta)
                
            log_p_grid[1] += self.log_cond_prob(left_run, left_run_cat, cat_dict, cat_count_dict, left_run_beta) + \
                             self.log_cond_prob(right_run, right_run_cat, cat_dict, cat_count_dict, right_run_beta, avoid_cat = left_run_cat)
                                                
            outcome = sample(a = grid, p = lognormalize(log_p_grid))
            
            if outcome == 1:
                # insert the new bundle
                bisect.insort(self.bundles, nth)
                # assign category 
                if right_run_cat:
                    self.categories.insert(original_idx, right_run_cat)
                    if right_run_cat not in self.beta: self.beta[right_run_cat] = right_run_beta
                else:
                    # random intial value - for convenience, just use a new category
                    # which will almost surely will be replaced
                    self.categories.insert(self.bundles.index(nth), new_cat)
                    if new_cat not in self.beta: self.beta[new_cat] = self.ibeta
                    # since we copied the category of the left run, its associated beta is already in self.beta

            self.total_time += time() - a_time
        return

    def batch_sample_beta(self):
        """Perform Metropolis Hastings sampling on beta.
        """
        # cdef things
        cdef int cat, cat_size
        cdef double old_beta, new_beta, proposal_sd, log_g_old, log_g_new, log_q_old, log_q_new, u, moving_prob
        
        # derive contexts from breakpoints arrangement
        cdef dict cat_dict = self.get_category_flat_dict()
        for cat in cat_dict.keys():
            
            old_beta = self.beta[cat]
            new_beta = -1
            proposal_sd = 0.1
            while new_beta <= 0:# or new_beta >= 1:
                new_beta = random.gauss(mu = old_beta, sigma = proposal_sd)

            # set up to calculate the g densities for both the old and new beta values
            log_g_old = -1 # * old_beta # which is np.log(np.exp(-1 * old_beta))
            log_g_new = -1 # * new_beta # similar as above

            cat_size = len(cat_dict[cat])
            
            log_g_old += lgamma(self.support_size * old_beta) \
                - lgamma(self.support_size * old_beta + cat_size)
            log_g_new += lgamma(self.support_size * new_beta) \
                - lgamma(self.support_size * new_beta + cat_size)

            for y in self.support:
                log_g_old += lgamma((cat_dict[cat] == y).sum() + old_beta) - lgamma(old_beta)
                log_g_new += lgamma((cat_dict[cat] == y).sum() + new_beta) - lgamma(new_beta)

            # compute candidate densities q for old and new beta
            # since the proposal distribution is normal this step is not needed
            log_q_old = 0
            log_q_new = 0
        
            # compute the moving probability
            moving_prob = (log_g_new + log_q_old) - (log_g_old + log_q_new)
            
            u = random.uniform(0,1)
            if log(u) < moving_prob: 
                self.beta[cat] = new_beta

    def batch_sample_l(self):
        """Perform Gibbs sampling on the mean length of a run.
        """
        total_number_of_runs = len(self.bundles)
        total_run_length = self.N
        
        if self.prior_type == 'Poisson':
            # we want to draw from Gamma(alpha + total_run_length, beta + total_number_of_runs)
            self.l = np.random.gamma(shape = self.poisson_prior_shape + total_run_length, 
                                     scale = 1. / (self.poisson_prior_rate + total_number_of_runs))
        elif self.prior_type == 'Geometric':
            # Beta(alpha + total_number_of_runs, beta + sum(all_run_lengths) - total_number_of_runs)
            self.l = np.random.beta(a = self.geom_prior_alpha + total_number_of_runs, 
                                    b = self.geom_prior_beta + total_run_length - total_number_of_runs)
            
    def batch_sample_categories(self):
        """Perform Gibbs sampling on the category of each bundle.
        """
        cdef int bundle_count, i, new_cat, cat, cat_index
        cdef dict cat_dict
        cdef list bundle_obs, cat_grid
        cdef np.ndarray y_count_arr, log_p_grid, uniq_cats, cat_count
        cdef double log_crp_prior, log_likelihood
        
        bundle_count = len(self.bundles)
        for i in xrange(bundle_count):
            # get all the observations in this bundle
            try: bundle_obs = list(self.data[self.bundles[i]:self.bundles[i+1]])
            except IndexError: bundle_obs = list(self.data[self.bundles[i]:])
            # count each support dim
            y_count_arr = np.array([bundle_obs.count(y) for y in self.support])

            # get a category - observations dict
            cat_dict = self.get_category_flat_dict(avoid = i)
            
            # get existing categories, novel category, and existing category counts
            try: cat_count, uniq_cats, new_cat = smallest_unused_label(self.categories[:i] + self.categories[i+1:])
            except IndexError: cat_count, uniq_cats, new_cat = self.smallest_unused_label(self.categories[:i])
            # set up grid
            cat_grid = list(uniq_cats) + [new_cat]
            log_p_grid = np.empty(len(cat_grid))
            
            for cat_index in xrange(len(cat_grid)):
                cat = cat_grid[cat_index]
                if cat == new_cat:
                    log_crp_prior = log(self.alpha / (bundle_count - 1 + self.alpha))
                    log_likelihood = len(bundle_obs) * log(1 / self.support_size)
                else:
                    log_crp_prior = log(cat_count[cat] / (bundle_count - 1 + self.alpha))
                    cat_n_arr = np.array([(cat_dict[cat] == y).sum() for y in self.support])
                    log_likelihood = (y_count_arr * np.log((cat_n_arr + self.beta[cat]) /
                                                          (cat_dict[cat].shape[0] + self.support_size * self.beta[cat]))).sum()
                
                log_p_grid[cat_index] = log_crp_prior + log_likelihood
            
            # sample new categories
            self.categories[i] = sample(a = cat_grid, p = lognormalize(log_p_grid))
            if self.categories[i] == new_cat: self.beta[new_cat] = self.ibeta

        return
        # address the label switching problem to some degree
        cat_dict = self.get_category_flat_dict()
        cat_support_count = np.empty(shape = (len(cat_dict.keys()), self.support_size))
        for cat in cat_dict.iterkeys():
            cat_idx = cat_dict.keys().index(cat)
            cat_support_count[cat_idx] = [-1 * (cat_dict[cat] == _).sum() for _ in self.support] # -1 for descending
        
        reindex = list(np.lexsort((eval(','.join(['cat_support_count[:,%s]' % _ for _ in xrange(self.support_size - 1, -1, -1)])))))
        categories_copy = np.array(self.categories)
        beta_copy = {}

        for cat in cat_dict.iterkeys():
            cat_idx = cat_dict.keys().index(cat)
            categories_copy[np.where(self.categories == cat)] = reindex.index(cat_idx) + 1
            beta_copy[reindex.index(cat_idx) + 1] = self.beta[cat]

        self.categories = list(categories_copy)
        self.beta = beta_copy
        return

    def log_length_prior(self, list runs):
        """Calculate the prior probability of a run, based on
        its length and its category
        """
        cdef list length_list = [float(len(_)) for _ in runs]
        cdef double l = float(self.l)
        
        # length-based prior
        if self.prior_type == 'Poisson':
            return np.array([log_dpois(_, l) for _ in length_list])#poisson.logpmf(length_list, self.l)
        elif self.prior_type == 'Geometric':
            return np.array([(run_length - 1) * log(1 - l) + log(l) for run_length in length_list])

    def log_cond_prob(self, list obs, cat, dict cat_dict, dict cat_count_dict, beta = None, avoid_cat = None):
        """Calculate the conditional probability of observations given category and beta.
        """
        cdef double log_p, prior, likelihood, p, alpha
        cdef int c, cat_N, num_bundles, support_size
        cdef np.ndarray[np.int_t, ndim = 1] cat_n_arr, y_count_arr

        alpha = self.alpha; num_bundles = len(self.categories)
        support_size = self.support_size
        
        log_p = 0
        y_count_arr = np.array([obs.count(y) for y in self.support])

        if cat is not None:
            try: cat_N = len(cat_dict[cat])
            except KeyError: cat_N = 0
            cat_n_arr = np.array([cat_count_dict[cat][y] if cat in cat_count_dict and y in cat_count_dict[cat] else 0 for y in self.support])
            log_p += (y_count_arr * np.log((cat_n_arr + beta) / (cat_N + support_size * beta))).sum()

        else:
            # If cat is None, then marginalize over all possible categories
            # the new bundle can take. This applies both when removing an 
            # existing breakpoint or adding a new breakpoint.
            p = 0
            for c in cat_dict.keys():
                if c == avoid_cat: continue
                cat_N = len(cat_dict[c])
                # prior 
                prior = self.categories.count(c) / (num_bundles + alpha)
                # likelihood
                cat_n_arr = np.array([cat_count_dict[cat][y] if cat in cat_count_dict and y in cat_count_dict[cat] else 0 for y in self.support])
                likelihood = (((cat_n_arr + self.beta[c]) / (cat_N + support_size * self.beta[c])) ** y_count_arr).prod()
                p += prior * likelihood
                
            # new category    
            prior = alpha / (num_bundles + alpha)
            likelihood = ((1 / support_size) ** y_count_arr).prod()
            p += prior * likelihood

            # convert to log
            if p > 0: log_p += log(p)

        return log_p

    def get_surround_runs(self, list bps, int target_bp, data=None):
        """Returns the left and right runs of a target point.
        """
        cdef list left_run, right_run
        if data is None: data = self.data
        cdef int anchor = bisect.bisect(bps, target_bp)

        try: left_run = list(data[bps[anchor-1]:target_bp])
        except: left_run = []

        if anchor == len(bps):
            right_run = list(data[target_bp:])
        else:
            right_run = list(data[target_bp:bps[anchor]])

        return left_run, right_run

    def get_category_flat_dict(self, avoid=None, bundles=None, categories=None, data=None):
        """Returns category-indexed obs.
        """
        cdef int i
        cdef np.ndarray run
        
        if data is None: data = self.data
        if bundles is None: bundles = self.bundles
        if categories is None: categories = self.categories

        cdef dict cat_flat_dict = {}
        for i in xrange(len(bundles)):
            if avoid is not None:
                if type(avoid) is list and i in avoid: continue
                elif i == avoid: continue
            try: run = data[bundles[i]:bundles[i+1]]
            except IndexError: run = data[bundles[i]:]
            try: cat_flat_dict[categories[i]] = np.hstack((cat_flat_dict[categories[i]], run))
            except KeyError: cat_flat_dict[categories[i]] = run
        return cat_flat_dict

    def _logprob(self, sample):
        """Calculate the joint probability of an HGPMM model structure and 
        data.
        """
        cdef int cat, bundle_length, y_count
        cdef double loglik, length_prior_logp, crp_prior_logp, l_logp, l, alpha, total_logp
        cdef list bundles, categories
        cdef dict beta
        cdef np.ndarray bundle_data
        
        bundles, categories, l, alpha, beta = sample

        cdef dict cat_flat_dict = {}
        cdef dict cat_bundle_count = {}
        cdef int N = 0, i


        # calculate the logp of l first
        if self.prior_type == 'Poisson':
            l_logp = log_dgamma(l, self.poisson_prior_shape, 1 / self.poisson_prior_rate)
        else:
            l_logp = beta_dist.logpdf(l, a = self.geom_prior_alpha, b = self.geom_prior_beta)

        total_logp = l_logp

        for i in xrange(len(bundles)):
                        
            if i == len(bundles) - 1:
                bundle_length = len(self.data) - bundles[i]
                bundle_data = self.data[bundles[i]:]
            else:
                bundle_length = bundles[i+1] - bundles[i]
                bundle_data = self.data[bundles[i]:bundles[i+1]]
                
            bundle_counter = Counter(bundle_data)
                
            # calculate the length prior
            if self.prior_type == 'Poisson':
                length_prior_logp = log_dpois(float(bundle_length), float(l))
            else:
                length_prior_logp = (bundle_length - 1) * log(1 - l) + log(l)
                
            # calculate the CPR prior, loglikleihood of data
            cat = categories[i]
            loglik = 0

            # if this is an existing category
            if cat in cat_flat_dict:
                crp_prior_logp = np.log(cat_bundle_count[cat] / (N + alpha))

                # loglik
                for y in self.support:
                    try: y_count = bundle_counter[y]
                    except KeyError: y_count = 0
                    loglik += y_count * np.log(((cat_flat_dict[cat] == y).sum() + beta[cat]) /
                                               (cat_flat_dict[cat].shape[0] + self.support_size * beta[cat]))
                
                # add this bundle to cluster_dict
                cat_bundle_count[cat] += 1
                cat_flat_dict[cat] = np.hstack((cat_flat_dict[cat], bundle_data))

            # if this is a new category
            else:
                crp_prior_logp = np.log(alpha / (N + alpha))

                # loglik
                for y in self.support:
                    try: y_count = bundle_counter[y]
                    except KeyError: y_count = 0
                    loglik += y_count * np.log(1 / self.support_size)

                # add this category
                cat_bundle_count[cat] = 1
                cat_flat_dict[cat] = bundle_data

            # use a simple count to avoid calculating N every time
            N += 1
                
            total_logp += length_prior_logp + crp_prior_logp + loglik

        return total_logp
    
    def run(self):
        """Run the sampler.
        """
        cdef str header
        if self.cutoff:
            beta_fp = gzip.open(self.source_dirname + 'beta-samples-' + self.source_filename + '-cutoff%d.csv.gz' % self.cutoff, 'w')
            sample_fp = gzip.open(self.source_dirname + 'bundle-samples-' + self.source_filename + '-cutoff%d.csv.gz' % self.cutoff, 'w')
            predict_fp = gzip.open(self.source_dirname + 'predictions-' + self.source_filename + '-cutoff%d.csv.gz' % self.cutoff, 'w')
        else:
            beta_fp = gzip.open(self.source_dirname + 'beta-samples-' + self.source_filename + '.csv.gz', 'w')
            sample_fp = gzip.open(self.source_dirname + 'bundle-samples-' + self.source_filename + '.csv.gz', 'w')
            predict_fp = gzip.open(self.source_dirname + 'predictions-' + self.source_filename + '.csv.gz', 'w')

        header = 'iteration,loglik,alpha,l,'
        header += ','.join([str(t) for t in xrange(1, self.N+1)])
        print(header, file = sample_fp)
        if self.output_to_stdout: print(header, file = sys.stdout)

        print('cutoff,iteration,category,beta', file=beta_fp)

        cdef int i
        
        # run the sampler
        for i in xrange(self.sample_size):
            self.iteration = i + 1
            self.set_temperature(self.iteration)
            self.batch_sample_bundles()
            self.batch_sample_l()
            self.batch_sample_categories()
            if self.sample_alpha: self.batch_sample_alpha()
            if self.sample_beta: self.batch_sample_beta()

            if self.record_best:
                if self.auto_save_sample((self.bundles, self.categories, self.l, self.alpha, self.beta)):
                    # save the samples to files
                    self.loglik = self.best_sample[1]
                    self.print_samples(iteration = self.iteration, dest = sample_fp)
                    for cat in np.unique(self.categories):
                        print(*[self.cutoff, self.iteration, cat, self.beta[cat]], sep=',', file=beta_fp)
                if self.no_improvement():
                    break
            else:
                # record the results for each iteration
                self.loglik = self._logprob((self.bundles, self.categories, self.l, self.alpha, self.beta))
                self.print_samples(iteration = self.iteration, dest = sample_fp)
                for cat in np.unique(self.categories):
                    print(*[self.cutoff, self.iteration, cat, self.beta[cat]], sep=',', file=beta_fp)

        # write out predictions
        if self.record_best:
            predicted = self.predict(self.best_sample[0])
            print(*['t', 'pos', 'probability'], sep=',', file=predict_fp)
            for i in xrange(self.support_size):
                print(*[self.N+1, self.support[i], predicted[i]], sep=',', file=predict_fp)
                
        # close files
        beta_fp.close()
        sample_fp.close()
        predict_fp.close()

        return self.total_time

    def print_samples(self, iteration, dest):
        """Print some debug information.
        """
        output = '{0},{1},{2},{3},'.format(iteration, self.loglik, self.alpha, self.l)

        # display category information
        cat_seq = []
        for i in xrange(len(self.bundles)):
            try: cat_seq.extend([self.categories[i]] * (self.bundles[i+1] - self.bundles[i]))
            except IndexError: cat_seq.extend([self.categories[i]] * (self.N - self.bundles[i]))
        output += ','.join([str(c) for c in cat_seq])
        print(output, file = dest)

        if self.output_to_stdout:
            print(output, file = sys.stdout)

    def predict(self, sample):
        """Calculate the probability of the entire support at the next time step
        given a sample.
        """
        bundles, categories, l, alpha, beta = sample
        cat_flat_dict = self.get_category_flat_dict(bundles=bundles, categories=categories)
        
        # initialize the final probability value
        p = np.zeros(self.support_size)
        
        # calculate a few probabilities regarding whether the current bundle stops or continues
        current_bundle_length = self.N - bundles[-1]
        p_bl_gtoeq_N = 1 - poisson.cdf(current_bundle_length - 1, l)
        p_bl_eq_N = np.exp(log_dpois(current_bundle_length, l))
        p_b_new = p_bl_eq_N / p_bl_gtoeq_N
        p_b_cont = 1 - p_b_new

        # case 1: bundle continues
        old_cat = categories[-1]
        for i in xrange(self.support_size):
            p[i] += p_b_cont * ((cat_flat_dict[old_cat] == self.support[i]).sum() + beta[old_cat]) / (cat_flat_dict[old_cat].shape[0] + self.support_size * beta[old_cat])
        # case 2: new bundle
        for cat in cat_flat_dict.iterkeys():
            if cat == old_cat: continue
            crp_prior_p = categories.count(cat) / (len(categories) + alpha)
            # loglik
            for i in xrange(self.support_size):
                likelihood = ((cat_flat_dict[cat] == self.support[i]).sum() + beta[cat]) / (cat_flat_dict[cat].shape[0] + self.support_size * beta[cat])
                p[i] += p_b_new * crp_prior_p * likelihood
            
        # if this is a new category
        crp_prior_p = alpha / (len(categories) + alpha)
        for i in xrange(self.support_size):
            likelihood = (1 / self.support_size)
            p[i] += p_b_new * crp_prior_p * likelihood

        p = p / p.sum()
        
        return p
        
