#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from BaseSampler import *
from scipy import stats
from datetime import datetime
import bisect
from collections import deque

class HGPMixtureSampler(BaseSampler):

    def __init__(self, data_file, sample_size, ialpha=1., ibeta=1., il = 0.5, s_type='batch', cutoff=None, annealing=False,
                 prior_type = 'Geometric', sample_output_file = sys.stdout, resample_rate = 0.33,
                 sample_alpha = True, sample_beta = True, use_context = False, sample_l = True, sample_category = True,
                 gamma_prior_shape = 1, gamma_prior_rate = 1,
                 geom_prior_alpha = 1., geom_prior_beta = 1.):
        """Initialize the constructor.
        """
        BaseSampler.__init__(self, data_file, sample_size, cutoff, annealing, sample_output_file)
        # other shared parameters
        self.s_type = s_type
        self.prior_type = prior_type
        self.sample_alpha = sample_alpha in ['T', 'True', True]
        self.sample_beta = sample_beta in ['T', 'True', True]
        self.sample_l = sample_l in ['T', 'True', True]
        self.sample_category = sample_category in ['T', 'True', True]
        self.use_context = use_context in ['T', 'True', True]
        self.resample_rate = resample_rate

        if self.prior_type == 'Poisson':
            # hyperpriors for self.l
            self.gamma_prior_shape = gamma_prior_shape
            self.gamma_prior_rate = gamma_prior_rate
        elif self.prior_type == 'Geometric':
            self.geom_prior_alpha = geom_prior_alpha
            self.geom_prior_beta = geom_prior_beta

        if s_type in ['batch']:
            self.categories = [1 for i in xrange(self.total_trial)]#np.zeros(self.total_trial, dtype='int16')
            #self.categories[0] = 1
            self.alpha = ialpha
            self.beta = ibeta
            if self.prior_type == 'Poisson':
                self.l = 5#np.random.gamma(shape = self.gamma_prior_shape, scale = 1. / self.gamma_prior_rate)
            elif self.prior_type == 'Geometric':
                self.l = il#np.random.beta(a = self.geom_prior_alpha, b = self.geom_prior_beta)

        elif s_type in ['increm']:
            self.categories = [[1 for _ in xrange(self.total_trial)] for _ in xrange(self.sample_size)]#np.empty((self.sample_size, self.total_trial), dtype='int8')
            self.alpha = np.empty(self.sample_size)
            self.alpha.fill(1.)
            self.beta = ibeta
            if self.prior_type == 'Poisson':
                self.l = np.random.gamma(shape = self.gamma_prior_shape, scale = 1 / (self.gamma_prior_rate),
                                         size = self.sample_size)
                self.l[np.where(self.l<1.)] = 1.
            elif self.prior_type == 'Geometric':
                self.l = np.random.beta(a = self.geom_prior_alpha, b = self.geom_prior_beta,
                                        size = self.sample_size)
            self.log_weight = np.log(np.ones(self.sample_size) / self.sample_size)

    def batch_sample_alpha(self, alpha, beta, l, categories, max_trial):
        """Perform Metropolis Hastings sampling on l.
        """
        old_alpha = alpha
        new_alpha = -1
        proposal_sd = 0.05#max(0.1, old_alpha)
        while new_alpha <= 0:
            new_alpha = random.gauss(mu = old_alpha, sigma = proposal_sd)
        
        # set up to calculate the g densities for both the old and new l values
        log_g_old = -1 * old_alpha # which is np.log(np.exp(-1 * old_alpha))
        log_g_new = -1 * new_alpha # similar as above

        # derive contexts from breakpoints arrangement
        for i in xrange(max_trial):
            target_cat = categories[i]
            existing_cats = categories[:i]
            try: cat_count = existing_cats.count(target_cat)
            except AttributeError: cat_count = np.where(existing_cats == target_cat)[0].size

            if cat_count == 0: 
                old_crp_prior = old_alpha / (len(existing_cats) + old_alpha)
                new_crp_prior = new_alpha / (len(existing_cats) + new_alpha)
            else: 
                old_crp_prior = cat_count / (len(existing_cats) + old_alpha)
                new_crp_prior = cat_count / (len(existing_cats) + new_alpha)
            
            change_prior = self.change_prior(categories = categories, at = i, c_l = l)

            if i == 0: previous_cat = categories[i]
            else: previous_cat = categories[i-1]
            log_g_old += np.log(change_prior * old_crp_prior + (1 - change_prior) * int(target_cat == previous_cat))
            log_g_new += np.log(change_prior * new_crp_prior + (1 - change_prior) * int(target_cat == previous_cat))

        # compute candidate densities q for old and new beta
        # since the proposal distribution is normal this step is not needed
        log_q_old = 0
        log_q_new = 0
        
        # compute the moving probability
        moving_log_prob = min(0, (log_g_new + log_q_old) - (log_g_old + log_q_new))
        
        u = random.uniform(0,1)
        if np.log(u) < moving_log_prob: return new_alpha
        else: return old_alpha

    def batch_sample_beta(self):
        """Perform Metropolis Hastings sampling on beta.
        """
        old_beta = self.beta
        new_beta = -1
        proposal_sd = 0.1#max(0.05, old_beta)
        while new_beta <= 0 or new_beta >= 1:
            new_beta = random.gauss(mu = old_beta, sigma = proposal_sd)

        # derive contexts from breakpoints arrangement
        cat_dict = self.make_category_dict()
        for cat in cat_dict.keys():

            # set up to calculate the g densities for both the old and new beta values
            log_g_old = -1 * old_beta # which is np.log(np.exp(-1 * old_beta))
            log_g_new = -1 * new_beta # similar as above

            log_g_old += math.lgamma(self.support_size * old_beta) \
                - math.lgamma(self.support_size * old_beta + len(cat_dict[cat]['data']))
            log_g_new += math.lgamma(self.support_size * new_beta) \
                - math.lgamma(self.support_size * new_beta + len(cat_dict[cat]['data']))
            
            for y in self.support:
                log_g_old += math.lgamma(cat_dict[cat]['data'].count(y) + old_beta) - math.lgamma(old_beta)
                log_g_new += math.lgamma(cat_dict[cat]['data'].count(y) + new_beta) - math.lgamma(new_beta)

        # compute candidate densities q for old and new beta
        # since the proposal distribution is normal this step is not needed
        log_q_old = 0
        log_q_new = 0
        
        # compute the moving probability
        moving_prob = min(1, np.exp((log_g_new + log_q_old) - (log_g_old + log_q_new)))
        #print(old_beta, log_g_old, new_beta, log_g_new, moving_prob)
        #raw_input()
        u = random.uniform(0,1)
        if u < moving_prob: self.beta = new_beta

    def batch_sample_l(self, alpha, beta, l, categories, max_trial):
        """Perform Metropolis Hastings sampling on l.
        """
        #total_number_of_runs = self.number_of_switches(categories)
        #total_run_length = self.total_trial
        #print(total_number_of_runs, total_run_length)
        #raw_input()
        #r = np.random.beta(a = self.geom_prior_alpha + total_number_of_runs, 
        #                   b = self.geom_prior_beta + total_run_length - total_number_of_runs)
        #return r

        old_l = l
        new_l = -1
        if self.prior_type == 'Geometric':
            proposal_sd = 0.1 #max(old_l * 0.5, 0.1)
            while new_l <= 0 or new_l >= 1:
                new_l = random.gauss(mu = old_l, sigma = proposal_sd)
                # set up to calculate the g densities for both the old and new l values
            log_g_old = stats.beta.logpdf(old_l, self.geom_prior_alpha, self.geom_prior_beta)
            log_g_new = stats.beta.logpdf(new_l, self.geom_prior_alpha, self.geom_prior_beta)

        elif self.prior_type == 'Poisson':
            proposal_sd = 1
            while new_l <= 1:
                new_l = random.gauss(mu = old_l, sigma = proposal_sd)

        for i in xrange(max_trial):
            target_cat = categories[i]
            existing_cats = categories[:i]
            try: cat_count = existing_cats.count(target_cat)
            except AttributeError: cat_count = np.where(existing_cats == target_cat)[0].size

            if cat_count == 0: crp_prior = alpha / (len(existing_cats) + alpha)
            else: crp_prior = cat_count / (len(existing_cats) + alpha)
            
            old_change_prior = self.change_prior(categories = categories, at = i, c_l = old_l)
            new_change_prior = self.change_prior(categories = categories, at = i, c_l = new_l)

            if i == 0: previous_cat = categories[i]
            else: previous_cat = categories[i-1]
            log_g_old += np.log(old_change_prior * crp_prior + (1 - old_change_prior) * int(target_cat == previous_cat))
            log_g_new += np.log(new_change_prior * crp_prior + (1 - new_change_prior) * int(target_cat == previous_cat))

        # compute candidate densities q for old and new beta
        # since the proposal distribution is normal this step is not needed
        log_q_old = 0
        log_q_new = 0 
        
        # compute the moving probability
        moving_log_prob = min(0, (log_g_new + log_q_old) - (log_g_old + log_q_new))
        
        u = random.uniform(0,1)
        if np.log(u) < moving_log_prob: return new_l
        else: return old_l

        #if self.prior_type == 'Poisson':
        #    # we want to draw from Gamma(alpha + total_run_length, beta + total_number_of_runs)
        #    self.l = np.random.gamma(shape = self.gamma_prior_shape + total_run_length, 
        #                             scale = 1. / (self.gamma_prior_rate + total_number_of_runs))

    def batch_sample_categories(self, alpha, beta, l, categories, max_trial):
        """Sample the categories of each event given other variables.
        """
        new_cats = copy.deepcopy(categories)

        for i in xrange(max_trial):
            cat_count, uniq_cats, new_cat = self.smallest_unused_label(new_cats[:i] + new_cats[i+1:max_trial])
            uniq_contexts = np.unique(np.hstack((self.context[:i], self.context[i+1:max_trial])))
            # make a category dictionary
            cat_dict = self.make_category_dict(avoid = i, categories = new_cats[:max_trial],
                                               data = self.data[:max_trial],
                                               context = self.context[:max_trial])

            # set up grid
            cat_grid = np.hstack((np.array(uniq_cats, dtype='int32'), new_cat))
            log_p_grid = np.empty(len(cat_grid))

            # take care of the changes to the next cat
            try: next_cat = new_cats[i+1]
            except IndexError: next_cat = None
            
            for cat_index in xrange(len(cat_grid)):
                cat = cat_grid[cat_index]
                if cat == new_cat:
                    crp_prior = alpha / (max_trial - 1. + alpha)
                    log_likelihood = np.log(beta / (self.support_size * beta))
                    if self.use_context:
                        log_likelihood += np.log(beta / (len(uniq_contexts) * beta))
                else:
                    crp_prior = len(cat_dict[cat]['data']) / (max_trial - 1. + alpha)
                    log_likelihood = np.log((cat_dict[cat]['data'].count(self.data[i]) + beta) / (len(cat_dict[cat]['data']) + self.support_size * beta))
                    if self.use_context:
                        log_likelihood += np.log((cat_dict[cat]['context'].count(self.context[i]) + beta) / 
                                                 (len(uniq_contexts) * beta + len(cat_dict[cat]['data'])))
               
                if i == 0:
                    change_prior = 1.
                else:
                    change_prior = self.change_prior(categories = new_cats, at = i, c_l = l)
                log_prior = np.log(change_prior * crp_prior + (1 - change_prior) * int(cat == new_cats[i-1])) 
                # also it seems this may contain a bug when i == 0, but since (1-change_prior) is 0, it doesn't matter.

                new_cats[i] = cat # implement it so that it's easier to get next_change_prior
                if next_cat is None: 
                    log_next_prior = 0
                else:
                    next_cat_count = cat_count[next_cat] + int(next_cat == cat)
                    #if cat_count[next_cat] >= 1: next_cat_count -= 1 # don't count itself
                    next_cat_count -= 1 # don't count itself
                    if next_cat_count == 0:
                        next_crp_prior = alpha / (max_trial - 1 + alpha)
                    else:
                        next_crp_prior = next_cat_count / (max_trial - 1 + alpha)
                    next_change_prior = self.change_prior(categories = new_cats, at = i+1, c_l = l)
                    log_next_prior = np.log(next_change_prior * next_crp_prior + (1 - next_change_prior) * int(cat == next_cat))

                log_p_grid[cat_index] = (log_prior + log_next_prior + log_likelihood) * self.temp
            
            new_cats[i] = np.random.choice(a = cat_grid, p = lognormalize(log_p_grid))
        return new_cats

    def increm_sample_everything(self, new_trial):
        """Perform sequential sampling on breakpoints.
        """
        # 1) propose new particles
        for i in xrange(self.sample_size):
            # 1.1) propose a new alpha value
            self.alpha[i] = self.batch_sample_alpha(alpha = self.alpha[i], beta = self.beta,
                                                    l = self.l[i], categories = self.categories[i],
                                                    max_trial = new_trial)
            # 1.2) propose a new l value
            self.l[i] = self.batch_sample_l(alpha = self.alpha[i], beta = self.beta,
                                            l = self.l[i], categories = self.categories[i],
                                            max_trial = new_trial)
            # 1.3) propose the category of the yet-to-be observed event
            if self.prior_type == 'Geometric':
                is_new_cluster = np.random.binomial(n = 1, p = self.l[i])
            
            if is_new_cluster == 0 & new_trial > 0: 
                self.categories[i][new_trial] = self.categories[i][new_trial - 1]
            else:
                # construct the CRP prior
                cat_count, _, new_cat = self.smallest_unused_label(self.categories[i][:new_trial])
                cat_count_dict = dict(enumerate(cat_count))
                # the above dict only stores counts of a category, not counts of observations within a category
                cat_count_dict[new_cat] = self.alpha[i] # this will create the 'new_cat' key even if it's not there yet
                try: del cat_count_dict[0]
                except: pass

                # normalize
                np_cat_counts = np.array(cat_count_dict.values())
                # sample the next value
                self.categories[i][new_trial] = np.random.choice(a = cat_count_dict.keys(),
                                                                 p = np_cat_counts / np_cat_counts.sum())

            # 1.4) weight the current configuration
            self.log_weight[i] += self.posterior_log_probability(max_trial = new_trial + 1,
                                                                 alpha = self.alpha[i], 
                                                                 beta = self.beta, 
                                                                 l = self.l[i], 
                                                                 categories = self.categories[i])

            #cat_dict = self.make_category_dict(categories = self.categories[i][:new_trial], 
            #                                   data = self.data[:new_trial])
            #target_cat = self.categories[i][new_trial]
            #if self.categories[i][new_trial] in cat_dict:
            #    self.log_weight[i] += np.log((cat_dict[target_cat].count(self.data[new_trial]) + self.beta) / \
            #        (len(cat_dict[target_cat]) + self.support_size * self.beta))
            #else:
            #    self.log_weight[i] += np.log(1. / self.support_size)

        self.log_weight = self.log_weight - np.max(self.log_weight)

        # step 3: resample breakpoints if necessary
        weights = lognormalize(self.log_weight)
        ess = 1 / (weights ** 2).sum()
        if ess < self.resample_rate * self.sample_size: 
            top_particle = np.where(weights == weights.max())[0][0]
            self.categories[0] = copy.deepcopy(self.categories[top_particle])
            self.l[0] = self.l[top_particle]
            self.alpha[0] = self.alpha[top_particle]
            #self.beta[0] = self.beta[top_particle]
            
            for i in xrange(1, self.sample_size):
                self.categories[i] = self.batch_sample_categories(alpha = self.alpha[i-1], beta = self.beta,
                                                                  l = self.l[i-1], 
                                                                  categories = self.categories[i-1],
                                                                  max_trial = new_trial+1)
                self.l[i] = self.batch_sample_l(alpha = self.alpha[i-1], beta = self.beta,
                                                l = self.l[i-1], categories = self.categories[i],
                                                max_trial = new_trial+1)
                self.alpha[i] = self.batch_sample_alpha(alpha = self.alpha[i-1], beta = self.beta,
                                                        l = self.l[i], categories = self.categories[i],
                                                        max_trial = new_trial+1)

            self.log_weight = np.ones(self.sample_size)
            self.log_weight = np.log(self.log_weight / np.sum(self.log_weight))

        return


    def change_prior(self, categories, at, c_l):
        """Calculate the prior probability of a change at trial i
        """
        # length-based prior
        if self.prior_type == 'Poisson':
            left_run_length = 0
            for i in xrange(at - 1, -1, -1):
                left_run_length += 1
                if categories[i] != categories[i-1]: break
            return stats.poisson.pmf(left_run_length, c_l) / (1. - stats.poisson.cdf(left_run_length - 1., c_l))
        elif self.prior_type == 'Geometric':
            return c_l

    def posterior_log_probability(self, max_trial, alpha, beta, l, categories):

        post_loglik = 0
        post_logjoint = 0
        for i in xrange(max_trial):
            target_cat = categories[i]
            existing_cats = categories[:i]
            cat_count = existing_cats.count(target_cat)
            uniq_contexts = np.unique(self.context[:i])

            ### calculate log prior ###
            # 1) get the CRP component
            if cat_count == 0: crp_prior = alpha / (len(existing_cats) + alpha)
            else: crp_prior = cat_count / (len(existing_cats) + alpha)
            # 2) get the change component
            change_prior = self.change_prior(categories = categories, at = i, c_l = l)
            # 3) do it
            if i == 0: previous_cat = categories[i]
            else: previous_cat = categories[i-1]
            log_prior = np.log(change_prior * crp_prior + (1 - change_prior) * int(target_cat == previous_cat))
       
            ### calculate log likelihood ###
            # 1) make a dictionary of categories
            cat_dict = self.make_category_dict(categories = existing_cats, 
                                               data = self.data[:i], context = self.context[:i])
            # 2) do it
            if target_cat in cat_dict:
                log_likelihood = np.log((cat_dict[target_cat]['data'].count(self.data[i]) + beta) / \
                                            (cat_count + self.support_size * beta))
                if self.use_context:
                    log_likelihood += np.log((cat_dict[target_cat]['context'].count(self.context[i]) + beta) / 
                                             (len(uniq_contexts) * beta + cat_count))
            else:
                log_likelihood = np.log(beta / (self.support_size * beta))
                if self.use_context:
                    log_likelihood += np.log(beta / max(beta, beta * len(uniq_contexts)))

            post_loglik += log_likelihood
            post_logjoint += log_prior + log_likelihood
        
        return post_loglik, post_logjoint

    def make_category_dict(self, avoid=None, categories=None, data=None, context=None):
        """Returns category-indexed obs.
        """
        if data is None: data = self.data
        if context is None: context = self.context
        if categories is None: categories = self.categories

        cat_dict = {}
        for i in xrange(len(categories)):
            if i == avoid: continue
            if categories[i] not in cat_dict: cat_dict[categories[i]] = {'data': [], 'context': []}
            #try: 
            cat_dict[categories[i]]['data'].append(data[i])
            #except KeyError: cat_dict[categories[i]]['data'] = [data[i]]
            #try: 
            if self.use_context: 
                cat_dict[categories[i]]['context'].append(context[i])
            #except KeyError: cat_dict[categories[i]]['context'] = [context[i]]

        return cat_dict

    def run(self):
        """Run the sampler.
        """
        if self.s_type == 'batch':
            header = 'loglik,logjoint,alpha,beta,l,'
            #header = 'alpha,l,'
            header += ','.join([str(t) for t in xrange(1, self.total_trial+1)])
            print(header, file = self.sample_output_file)

            for i in xrange(self.sample_size):
                self.iteration = i + 1
                if self.iteration % 50 == 0:
                    print('Iteration:', self.iteration, self.beta, file=sys.stderr)
                    if self.sample_output_file != sys.stdout: self.sample_output_file.flush()

                self.set_temperature()
                if self.sample_alpha:
                    self.alpha = self.batch_sample_alpha(alpha = self.alpha, beta = self.beta, 
                                                         l = self.l, categories = self.categories,
                                                         max_trial = self.total_trial)
                if self.sample_l:
                    self.l = self.batch_sample_l(alpha = self.alpha, beta = self.beta,
                                                 l = self.l, categories = self.categories,
                                                 max_trial = self.total_trial)
                if self.sample_category:
                    self.categories = self.batch_sample_categories(alpha = self.alpha, beta = self.beta,
                                                                   l = self.l, categories = self.categories,
                                                                   max_trial = self.total_trial)
                if self.sample_beta:
                    self.batch_sample_beta()
                
                self.print_batch_iteration(dest = self.sample_output_file)

        elif self.s_type == 'increm':
            # currently we output samples and predictions at the same time
            headers = 'trial,sample,weight,alpha,beta,l,'
            headers += ','.join([str(t) for t in xrange(1, self.total_trial+1)])
            print(headers, file=self.sample_output_file)
            
            self.increm_predict(trial = 0)
            
            for i in xrange(self.total_trial):
                self.iteration = i + 1
                self.set_temperature()
                self.increm_sample_everything(new_trial = i)

                try: self.increm_predict(trial = i+1)
                except IndexError: pass

                weights = lognormalize(self.log_weight)
                # debug info
                if i % 20 == 0: print('Trial:', i, file=sys.stderr)
                if i % 100 == 0: self.sample_output_file.flush()

                self.print_increm_iteration(dest = self.sample_output_file)

        if self.sample_output_file != sys.stdout: self.sample_output_file.close()

    def increm_predict(self, trial):
        
        if trial == 0: 
            print('trial', 'pos', 'nseg.pred', 'nseg.pred.se', sep=',', file=sys.stdout)
            print(trial+1, self.data[trial], self.beta / (self.beta * self.support_size), 0.0, sep=',', file=sys.stdout)
            return

        pred_p = np.zeros(self.sample_size)
        for p in xrange(self.sample_size):
            cat_count, uniq_cats, new_cat = self.smallest_unused_label(self.categories[p][:trial])
            cat_dict = self.make_category_dict(categories = self.categories[p][:trial],
                                               data = self.data[:trial],
                                               context = self.context[:trial])
            prev_cat = self.categories[p][trial - 1]
            cat_options = np.hstack((uniq_cats, new_cat))
            if self.prior_type == 'Geometric':
                change_prior = self.l[p]
            for cat in cat_options:
                if cat == new_cat:
                    prior = change_prior * self.alpha[p] / (trial + self.alpha[p])
                    likelihood = self.beta / (self.beta * self.support_size)
                    pred_p[p] += prior * likelihood
                else:
                    prior = change_prior * cat_count[cat] / (trial + self.alpha[p]) + (1 - change_prior) * int(cat == prev_cat)
                    likelihood = (cat_dict[cat]['data'].count(self.data[trial]) + self.beta) / (cat_count[cat] + self.beta * self.support_size)
                    pred_p[p] += prior * likelihood

        pred_p_point = np.dot(pred_p, lognormalize(self.log_weight))
        pred_p_se = pred_p.std() / np.sqrt(pred_p.size)
        print(trial+1, self.data[trial], pred_p_point.round(decimals=7), pred_p_se.round(decimals=7), sep=',', file=sys.stdout)

    def print_batch_iteration(self, dest):
        """Print some debug information.
        """
        post_loglik, post_logjoint = self.posterior_log_probability(alpha = self.alpha, beta = self.beta,
                                                                    l = self.l, categories = self.categories,
                                                                    max_trial = self.total_trial)

        # display beta information
        output = str(post_loglik) + ','
        output += str(post_logjoint) + ','
        output += str(self.alpha) + ','
        output += str(self.beta) + ','
        output += str(self.l) + ','
        # display category information
        output += ','.join([str(c) for c in self.reorder_labels(self.categories)])
        print(output, file = dest)

    def print_increm_iteration(self, dest):
        
        weights = lognormalize(self.log_weight)
        
        for p in xrange(self.sample_size):
            print(self.iteration, p, weights[p].round(decimals = 5), 
                  self.alpha[p].round(decimals = 5), self.beta, 
                  self.l[p].round(decimals = 5), 
                  *self.reorder_labels(self.categories[p][:self.iteration]), 
                  sep=',', file=self.sample_output_file)


    def reorder_labels(self, labels):
        labels = np.array(labels)
        cur_labels = uniqify(labels[np.where(labels > 0)])
        new_labels = range(1,len(cur_labels) + 1)
        labels_copy = copy.deepcopy(labels)
        for i in xrange(len(cur_labels)):
            labels_copy[np.where(labels == cur_labels[i])] = new_labels[i]
        return labels_copy

    def number_of_switches(self, lst):
        #print(lst)
        if len(lst) == 0: return 0
        last = lst[0]
        ns = 0
        for i in xrange(1, len(lst)):
            if lst[i] != last: 
                ns+=1
                last = lst[i]
        return ns
