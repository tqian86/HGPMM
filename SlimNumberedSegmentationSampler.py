#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
from BaseSampler import *
from scipy.stats import poisson
from datetime import datetime
import bisect

class SlimNumberedSegmentationSampler(BaseSampler):

    def __init__(self, data_file, sample_size, ialpha = .33, ibeta=1, s_type='batch', cutoff=None, annealing=False,
                 prior_type = 'Geometric', sample_output_file = sys.stdout, resample_rate = 0.33,
                 sample_alpha = True, sample_beta = True, use_context = False,
                 gamma_prior_shape = 1, gamma_prior_rate = 1,
                 geom_prior_alpha = 1, geom_prior_beta = 1):
        """Initialize the constructor.
        """
        BaseSampler.__init__(self, data_file, sample_size, cutoff, annealing, sample_output_file)
        # other shared parameters
        self.s_type = s_type
        self.prior_type = prior_type
        self.sample_alpha = sample_alpha in ['T', 'True', True]
        self.sample_beta = sample_beta in ['T', 'True', True]
        self.use_context = use_context in ['T', 'True', True]
        self.resample_rate = resample_rate
        self.ibeta = ibeta

        if self.prior_type == 'Poisson':
            # hyperpriors for self.l
            self.gamma_prior_shape = gamma_prior_shape
            self.gamma_prior_rate = gamma_prior_rate
        elif self.prior_type == 'Geometric':
            self.geom_prior_alpha = geom_prior_alpha
            self.geom_prior_beta = geom_prior_beta

        if s_type in ['batch']:
            self.bundles = [0]
            self.categories = [1]
            self.alpha = ialpha
            self.beta = {1:ibeta}
            if self.prior_type == 'Poisson':
                self.l = np.random.gamma(shape = self.gamma_prior_shape, scale = 1. / self.gamma_prior_rate)
            elif self.prior_type == 'Geometric':
                self.l = 0.5#np.random.beta(a = self.geom_prior_alpha, b = self.geom_prior_beta)

        elif s_type in ['increm']:
            self.breakpoints = np.zeros((self.sample_size, self.N), dtype='int8')
            self.breakpoints[:,0] = 1
            self.beta = ibeta
            if self.prior_type == 'Poisson':
                self.l = np.random.gamma(shape = self.gamma_prior_shape, scale = 1 / (self.gamma_prior_rate),
                                         size = self.sample_size)
                self.l[np.where(self.l<1.)] = 1.
            elif self.prior_type == 'Geometric':
                self.l = np.random.beta(a = self.geom_prior_alpha, b = self.geom_prior_beta,
                                        size = self.sample_size)
            self.log_weight = np.log(np.ones(self.sample_size) / self.sample_size)

            
    def batch_sample_bundles(self):
        """Perform Gibbs sampling on clusters.
        """
        # we index the sequence of observations using "nth"
        for nth in xrange(1, self.N):
            cat_dict = self.get_category_flat_dict()
            right_run_cat = None
            if nth in self.bundles:
                original_idx = self.bundles.index(nth)

                # get the left run
                left_run_cat = self.categories[original_idx-1]
                left_run_beta = self.beta[left_run_cat]
                left_run = self.data[self.bundles[original_idx-1]:self.bundles[original_idx]]

                # get the right run
                right_run_cat = self.categories[original_idx]
                right_run_beta = self.beta[right_run_cat]
                try: right_run = self.data[self.bundles[original_idx]:self.bundles[original_idx+1]]
                except: right_run = self.data[self.bundles[original_idx]:]
                
                together_beta = left_run_beta

                # remove
                del self.bundles[original_idx]
                del self.categories[original_idx]
                if self.categories.count(right_run_cat) == 0:
                    del self.beta[right_run_cat]
            else:
                left_run_cat = self.categories[bisect.bisect(self.bundles, nth)-1]
                left_run_beta = self.beta[left_run_cat]
                right_run_beta = left_run_beta
                together_beta = left_run_beta

                # get the left and right runs of a breakpoint
                left_run, right_run = self.get_surround_runs(bps = self.bundles, target_bp = nth)

                
            # set up the grid
            grid = [0, 1]
            log_p_grid = [0.] * len(grid)
            #print('bundles:', self.bundles, file=sys.stderr)
            #print('left:', left_run, file=sys.stderr)
            #print('right:', right_run, file=sys.stderr)

            # compute the prior probability of each run
            log_p_grid[0] = self.log_length_prior(runs = [left_run + right_run])[0] * self.temp
            log_p_grid[1] = (self.log_length_prior(runs = [left_run])[0] + \
                             self.log_length_prior(runs = [right_run])[0]) * self.temp
            
            # compute the likelihood of each case
            log_p_grid[0] += self.log_cond_prob(obs = left_run + right_run, cat = left_run_cat,
                                                cat_dict = cat_dict, beta = together_beta)
            log_p_grid[1] += self.log_cond_prob(left_run, left_run_cat, cat_dict, left_run_beta) + \
                             self.log_cond_prob(right_run, right_run_cat, cat_dict, right_run_beta, avoid_cat = left_run_cat)
                                                

            #log_p_grid[0] += self.log_joint_prob(obs = left_run + right_run, beta = together_beta)
            #log_p_grid[1] += self.log_joint_prob(obs = left_run, beta = left_run_beta) + \
            #    self.log_joint_prob(obs = right_run, beta = right_run_beta)
            
            #print(left_run_beta, right_run_beta, together_beta)
            #print(lognormalize(log_p_grid))
            #raw_input()

            outcome = np.random.choice(a = grid, p = lognormalize(log_p_grid))

            if outcome == 1:
                # insert the new bundle
                bisect.insort(self.bundles, nth)
                # assign category 
                if right_run_cat:
                    self.categories.insert(original_idx, right_run_cat)
                    if right_run_cat not in self.beta: self.beta[right_run_cat] = right_run_beta
                else:
                    self.categories.insert(self.bundles.index(nth), self.categories[self.bundles.index(nth)-1])
                    # since we copied the category of the left run, its associated beta is already in self.beta

        return

    def batch_sample_beta(self):
        """Perform Metropolis Hastings sampling on beta.
        """
        # derive contexts from breakpoints arrangement
        cat_dict = self.get_category_flat_dict()
        for cat in cat_dict.keys():
            
            old_beta = self.beta[cat]
            new_beta = -1
            proposal_sd = 0.1
            while new_beta <= 0:# or new_beta >= 1:
                new_beta = random.gauss(mu = old_beta, sigma = proposal_sd)

            # set up to calculate the g densities for both the old and new beta values
            log_g_old = -1 # * old_beta # which is np.log(np.exp(-1 * old_beta))
            log_g_new = -1 # * new_beta # similar as above

            log_g_old += math.lgamma(self.support_size * old_beta) \
                - math.lgamma(self.support_size * old_beta + len(cat_dict[cat]))
            log_g_new += math.lgamma(self.support_size * new_beta) \
                - math.lgamma(self.support_size * new_beta + len(cat_dict[cat]))
            
            for y in self.support:
                log_g_old += math.lgamma(cat_dict[cat].count(y) + old_beta) - math.lgamma(old_beta)
                log_g_new += math.lgamma(cat_dict[cat].count(y) + new_beta) - math.lgamma(new_beta)

            # compute candidate densities q for old and new beta
            # since the proposal distribution is normal this step is not needed
            log_q_old = 0
            log_q_new = 0
        
            # compute the moving probability
            moving_prob = (log_g_new + log_q_old) - (log_g_old + log_q_new)
            
            u = random.uniform(0,1)
            if np.log(u) < moving_prob: 
                self.beta[cat] = new_beta

    def batch_sample_l(self):
        """Perform Gibbs sampling on the mean length of a run.
        """
        total_number_of_runs = len(self.bundles)
        total_run_length = self.N
        
        if self.prior_type == 'Poisson':
            # we want to draw from Gamma(alpha + total_run_length, beta + total_number_of_runs)
            self.l = np.random.gamma(shape = self.gamma_prior_shape + total_run_length, 
                                     scale = 1. / (self.gamma_prior_rate + total_number_of_runs))
        elif self.prior_type == 'Geometric':
            # Beta(alpha + total_number_of_runs, beta + sum(all_run_lengths) - total_number_of_runs)
            self.l = np.random.beta(a = self.geom_prior_alpha + total_number_of_runs, 
                                    b = self.geom_prior_beta + total_run_length - total_number_of_runs)
            
    def batch_sample_categories(self):
        """Perform Gibbs sampling on the category of each bundle.
        """
        bundle_count = len(self.bundles)
        for i in xrange(bundle_count):
            # get all the observations in this bundle
            try: bundle_obs = self.data[self.bundles[i]:self.bundles[i+1]]
            except IndexError: bundle_obs = self.data[self.bundles[i]:]
            # get a category - observations dict
            cat_dict = self.get_category_flat_dict(avoid = i)
            
            # get existing categories, novel category, and existing category counts
            try: cat_count, uniq_cats, new_cat = self.smallest_unused_label(self.categories[:i] + self.categories[i+1:])
            except IndexError: cat_count, uniq_cats, new_cat = self.smallest_unused_label(self.categories[:i])
            # set up grid
            cat_grid = list(uniq_cats) + [new_cat]
            log_p_grid = np.empty(len(cat_grid))
            
            for cat in cat_grid:
                cat_index = cat_grid.index(cat)
                if cat == new_cat:
                    log_crp_prior = np.log(self.alpha / (bundle_count - 1 + self.alpha))
                    log_likelihood = len(bundle_obs) * np.log(1 / self.support_size)
                else:
                    log_crp_prior = np.log(cat_count[cat] / (bundle_count - 1 + self.alpha))
                    log_likelihood = 0
                    for y in self.support:
                        y_count = bundle_obs.count(y)
                        log_likelihood += y_count * np.log((cat_dict[cat].count(y) + self.beta[cat]) / (len(cat_dict[cat]) + self.support_size * self.beta[cat]))
                
                log_p_grid[cat_index] = log_crp_prior + log_likelihood
            
            # sample new categories
            self.categories[i] = np.random.choice(a = cat_grid, p = lognormalize(log_p_grid))
            if self.categories[i] == new_cat: self.beta[new_cat] = self.ibeta

    def log_length_prior(self, runs=None, length_list=None, c_l=None):
        """Calculate the prior probability of a run, based on
        its length and its category
        """
        if c_l is None: c_l = self.l
        if runs is None and length_list is None: 
            raise NameError('No length or run specified')

        if runs is not None: length_list = [len(_) for _ in runs]
        # length-based prior
        if self.prior_type == 'Poisson':
            return poisson.logpmf(length_list, c_l)
        elif self.prior_type == 'Geometric':
            return np.array([(run_length - 1) * np.log(1 - c_l) + np.log(c_l) for run_length in length_list])

    def log_joint_prob(self, obs, beta):
        """Calculate the joint probability of all observations of the same category,
        which may contain several runs.
        """
        log_p = math.lgamma(beta * self.support_size) - math.lgamma(beta * self.support_size + len(obs)) 
        for y in self.support:
            log_p += math.lgamma(obs.count(y) + beta) - math.lgamma(beta)
        return log_p

    def log_cond_prob(self, obs, cat, cat_dict, beta, avoid_cat = None):
        """Calculate the conditional probability of observations given category and beta.
        """
        log_p = 0
        if cat is not None:
            for y in self.support:
                y_count = obs.count(y)
                log_p += y_count * np.log((cat_dict[cat].count(y) + beta) / (len(cat_dict[cat]) + self.support_size * beta))
        else:
            p = 0
            for c in cat_dict.keys():
                if c == avoid_cat: continue
                log_prior = np.log(self.categories.count(c) / (len(self.categories) + self.alpha))
                log_lik = 0
                for y in self.support:
                    y_count = obs.count(y)
                    log_lik += y_count * np.log((cat_dict[c].count(y) + self.beta[c]) / (len(cat_dict[c]) + self.support_size * self.beta[c]))
                p += np.exp(log_prior + log_lik)
            # new category    
            log_prior = np.log(self.alpha / (len(self.categories) + self.alpha))
            log_lik = 0
            for y in self.support:
                y_count = obs.count(y)
                log_lik += y_count * np.log(1 / self.support_size)
            p += np.exp(log_prior + log_lik)
            if p > 0: log_p += np.log(p)
            
        return log_p

    def get_surround_runs(self, bps, target_bp, data=None):
        """Returns the left and right runs of a target point.
        """
        if data is None: data = self.data
        anchor = bisect.bisect(bps, target_bp)

        try: left_run = data[bps[anchor-1]:target_bp]
        except: left_run = []

        if anchor == len(bps):
            right_run = data[target_bp:]
        else:
            right_run = data[target_bp:bps[anchor]]

        return left_run, right_run

    def get_category_flat_dict(self, avoid=None, bundles=None, categories=None, data=None):
        """Returns category-indexed obs.
        """
        if data is None: data = self.data
        if bundles is None: bundles = self.bundles
        if categories is None: categories = self.categories

        cat_flat_dict = {}
        for i in xrange(len(bundles)):
            if avoid is not None and i == avoid: continue
            try: run = data[bundles[i]:bundles[i+1]]
            except IndexError: run = data[bundles[i]:]
            try: cat_flat_dict[categories[i]].extend(run)
            except KeyError: cat_flat_dict[categories[i]] = run
        return cat_flat_dict

    def run(self):
        """Run the sampler.
        """
        if self.s_type == 'batch':
            #header = 'alpha,beta,l,'
            header = 'alpha,l,'
            header += ','.join([str(t) for t in xrange(1, self.N+1)])
            print(header, file = self.sample_output_file)

            for i in xrange(self.sample_size):
                self.iteration = i + 1
                if self.iteration % 50 == 0:
                    if self.sample_output_file != sys.stdout: self.sample_output_file.flush()

                self.set_temperature()
                self.batch_sample_bundles()
                self.batch_sample_l()
                self.batch_sample_categories()
                if self.sample_beta: self.batch_sample_beta()
                # record the results
                self.print_batch_iteration(dest = self.sample_output_file)

        if self.sample_output_file != sys.stdout: self.sample_output_file.close()

    def increm_predict_next_trial(self, next_bp):
        
        if next_bp == 0: 
            print('trial.no', 'pos', 'nseg.pred', 'nseg.pred.se', sep=',', file=sys.stdout)
            print(next_bp+1, self.data[next_bp], self.beta / (self.beta * self.support_size), 0.0, sep=',', file=sys.stdout)
            return

        pred_p = np.zeros(self.sample_size)
        for p in xrange(self.sample_size):
            left_run, _, left_number, _ = self.get_surround_runs(bps = self.breakpoints[p][:next_bp], target_bp = next_bp, data = self.data[:next_bp])
            all_runs = self.get_categories(bps = self.breakpoints[p][:next_bp], data = self.data[:next_bp])
            next_run_numbers = [-1, 0] + all_runs.keys()
            if left_number is not None: 
                next_run_numbers.remove(left_number)
                left_number_count = np.where(self.breakpoints[p] == left_number)[0].size
            else: left_number_count = 0
            if self.prior_type == 'Poisson':
                change_prior = poisson.pmf(len(left_run), self.l[p]) / (1. - poisson.cdf(len(left_run) - 1., self.l[p]))
                if change_prior == float('Inf'): change_prior = 1.
            elif self.prior_type == 'Geometric':
                change_prior = self.l[p]
            for number in next_run_numbers:
                if number == 0:
                    pred_p[p] += (1 - change_prior) * (all_runs[left_number].count(self.data[next_bp]) + self.beta) / (len(all_runs[left_number]) + self.beta * self.support_size)
                elif number == -1:
                    pred_p[p] += change_prior * (self.alpha / (self.alpha + np.nonzero(self.breakpoints[p])[0].size - left_number_count)) * \
                        (self.beta / (self.beta * self.support_size))
                else:
                    pred_p[p] += change_prior * \
                        (np.where(self.breakpoints[p] == number)[0].size / (self.alpha + np.nonzero(self.breakpoints[p])[0].size - left_number_count)) * \
                        (all_runs[number].count(self.data[next_bp]) + self.beta) / (len(all_runs[number]) + self.beta * self.support_size)
                #print('bp:', next_bp, self.data[next_bp], 'candidate:', number, file=sys.stderr)
                #print('total run:', np.nonzero(self.breakpoints[p])[0].size, 'left:', left_number, left_number_count, change_prior, pred_p[p], file=sys.stderr)
                #raw_input()

        pred_p_point = np.dot(pred_p, lognormalize(self.log_weight))
        pred_p_se = pred_p.std() / np.sqrt(pred_p.size)
        print(next_bp+1, self.data[next_bp], pred_p_point.round(decimals=7), pred_p_se.round(decimals=7), sep=',', file=sys.stdout)

    def print_batch_iteration(self, dest):
        """Print some debug information.
        """
        # display beta information
        output = str(self.alpha) + ','
        #output += str(self.beta) + ','
        output += str(self.l) + ','
        # display category information
        cat_seq = []
        for i in xrange(len(self.bundles)):
            try: cat_seq.extend([self.categories[i]] * (self.bundles[i+1] - self.bundles[i]))
            except IndexError: cat_seq.extend([self.categories[i]] * (self.N - self.bundles[i]))
        output += ','.join([str(c) for c in cat_seq])
        print(output, file = dest)

    def reorder_labels(self, labels):
        labels = np.array(labels)
        cur_labels = list(set(labels[np.where(labels > 0)]))
        new_labels = range(1,len(cur_labels) + 1)
        labels_copy = copy.deepcopy(labels)
        for i in xrange(len(cur_labels)):
            labels_copy[np.where(labels == cur_labels[i])] = new_labels[i]
        return labels_copy
