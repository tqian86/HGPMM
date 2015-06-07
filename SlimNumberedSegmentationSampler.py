#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from BaseSampler import *
from scipy.stats import poisson
from datetime import datetime
import bisect

class SlimNumberedSegmentationSampler(BaseSampler):

    def __init__(self, data_file, sample_size, ialpha = .33, ibeta=1., s_type='batch', cutoff=None, annealing=False,
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

        if self.prior_type == 'Poisson':
            # hyperpriors for self.l
            self.gamma_prior_shape = gamma_prior_shape
            self.gamma_prior_rate = gamma_prior_rate
        elif self.prior_type == 'Geometric':
            self.geom_prior_alpha = geom_prior_alpha
            self.geom_prior_beta = geom_prior_beta

        if s_type in ['batch']:
            self.clusters = [0]            
            self.categories = [1]
            self.alpha = ialpha
            self.beta = [ibeta]
            if self.prior_type == 'Poisson':
                self.l = np.random.gamma(shape = self.gamma_prior_shape, scale = 1. / self.gamma_prior_rate)
            elif self.prior_type == 'Geometric':
                self.l = 0.5#np.random.beta(a = self.geom_prior_alpha, b = self.geom_prior_beta)

        elif s_type in ['increm']:
            self.breakpoints = np.zeros((self.sample_size, self.total_trial), dtype='int8')
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

    def batch_sample_clusters(self):
        """Perform Gibbs sampling on clusters.
        """
        for i in xrange(1, self.total_trial):
            existing = False
            if i in self.clusters:
                existing = True
                original_cat = self.categories[self.clusters.index(i)]
                left_run_beta = self.beta[self.clusters.index(i)-1]
                right_run_beta = self.beta[self.clusters.index(i)]
                together_beta = left_run_beta
                del self.categories[self.clusters.index(i)]
                del self.beta[self.clusters.index(i)]
                self.clusters.remove(i)
            else:
                left_run_beta = self.beta[bisect.bisect(self.clusters, i)-1]
                right_run_beta = left_run_beta
                together_beta = left_run_beta

            # set up the grid
            grid = [1, 0]
            log_p_grid = [0.] * len(grid)
            # get the left and right runs of a breakpoint
            left_run, right_run = self.get_surround_runs(bps = self.clusters, target_bp = i)
            #print('clusters:', self.clusters, file=sys.stderr)
            #print('left:', left_run, file=sys.stderr)
            #print('right:', right_run, file=sys.stderr)

            # compute the prior probability of each run
            log_p_grid[0] = (self.log_length_prior(runs = [left_run])[0] + \
                             self.log_length_prior(runs = [right_run])[0]) * self.temp
            log_p_grid[1] = self.log_length_prior(runs = [left_run + right_run])[0] * self.temp
            # compute the likelihood of each case
            log_p_grid[0] += self.log_joint_prob(obs = left_run, beta = left_run_beta) + \
                self.log_joint_prob(obs = right_run, beta = right_run_beta)
            log_p_grid[1] += self.log_joint_prob(obs = left_run + right_run, beta = together_beta)
            
            #print(left_run_beta, right_run_beta, together_beta)
            #print(lognormalize(log_p_grid))
            #raw_input()

            outcome = np.random.choice(a = grid, p = lognormalize(log_p_grid))

            #print(outcome, file=sys.stderr)
            if outcome == 1:
                # insert the new cluster
                bisect.insort(self.clusters, i)
                bisect.insort(self.beta, right_run_beta)
                # assign an initial category market
                if existing:
                    self.categories.insert(self.clusters.index(i), original_cat)
                else:
                    _, _, new_cat = self.smallest_unused_label(self.categories)
                    self.categories.insert(self.clusters.index(i), self.categories[self.clusters.index(i)-1])
                    #self.categories.insert(self.clusters.index(i), new_cat)
            
        return

    def batch_sample_beta(self):
        """Perform Metropolis Hastings sampling on beta.
        """

        # derive contexts from breakpoints arrangement
        cat_dict = self.get_category_flat_dict()
        for cat in cat_dict.keys():
            
            beta_indices = np.where(np.array(self.categories) == cat)[0]
            #print(beta_indices, cat, self.categories)
            #sys.exit(0)
            old_beta = self.beta[beta_indices[0]]
            new_beta = -1
            proposal_sd = max(0.1, old_beta)
            while new_beta <= 0:# or new_beta >= 1:
                new_beta = random.gauss(mu = old_beta, sigma = proposal_sd)
            #new_beta = np.random.choice([0.00001, .1])

            # set up to calculate the g densities for both the old and new beta values
            log_g_old = -1# * old_beta # which is np.log(np.exp(-1 * old_beta))
            log_g_new = -1# * new_beta # similar as above

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
            moving_prob = min(1, np.exp((log_g_new + log_q_old) - (log_g_old + log_q_new)))
            
            u = random.uniform(0,1)
            if u < moving_prob: 
                for bi in beta_indices:
                    self.beta[bi] = new_beta


    def batch_sample_l(self):
        """Perform Gibbs sampling on the mean length of a run.
        """
        total_number_of_runs = len(self.clusters)
        total_run_length = self.total_trial
        
        if self.prior_type == 'Poisson':
            # we want to draw from Gamma(alpha + total_run_length, beta + total_number_of_runs)
            self.l = np.random.gamma(shape = self.gamma_prior_shape + total_run_length, 
                                     scale = 1. / (self.gamma_prior_rate + total_number_of_runs))
        elif self.prior_type == 'Geometric':
            # Beta(alpha + total_number_of_runs, beta + sum(all_run_lengths) - total_number_of_runs)
            self.l = np.random.beta(a = self.geom_prior_alpha + total_number_of_runs, 
                                    b = self.geom_prior_beta + total_run_length - total_number_of_runs)
            
    def tversky_sample_l(self):
        run_lengths = np.diff(self.clusters + [self.total_trial])

        l_grid = range(1,22)
        l_log_p_grid = np.zeros(len(l_grid))

        if self.prior_type == 'Poisson':
            for i in xrange(len(l_grid)):
                l_log_p_grid[i] = poisson.logpmf(run_lengths, l_grid[i]).sum()
            #print(l_log_p_grid)
            self.l = np.random.choice(a = l_grid, p = lognormalize(l_log_p_grid))

        elif self.prior_type == 'Geometric':
            # Beta(alpha + total_number_of_runs, beta + sum(all_run_lengths) - total_number_of_runs)
            self.l = np.random.beta(a = self.geom_prior_alpha + total_number_of_runs, 
                                    b = self.geom_prior_beta + total_run_length - total_number_of_runs)
        

    def batch_sample_categories(self):
        """Perform Gibbs sampling on the category of each cluster.
        """
        
        cluster_count = len(self.clusters)
        for i in xrange(cluster_count):
            # get all the observations in this cluster
            try: cluster_obs = self.data[self.clusters[i]:self.clusters[i+1]]
            except IndexError: cluster_obs = self.data[self.clusters[i]:]
            # get a category - observations dict
            cat_dict = self.get_category_flat_dict(avoid = i)
            
            # get existing categories, novel category, and existing category counts
            try: cat_count, uniq_cats, new_cat = self.smallest_unused_label(self.categories[:i] + self.categories[i+1:])
            except IndexError: cat_count, uniq_cats, new_cat = self.smallest_unused_label(self.categories[:i])
            # set up grid
            cat_grid = np.hstack((uniq_cats, new_cat))
            log_p_grid = np.empty(len(cat_grid))

            for cat_index in xrange(len(cat_grid)):
                cat = cat_grid[cat_index]
                #print(cat, cat_dict.keys(), file=sys.stderr)
                if cat == new_cat:
                    log_crp_prior = np.log(self.alpha / (cluster_count - 1. + self.alpha))
                    log_likelihood = len(cluster_obs) * np.log(self.beta[i] / (self.support_size * self.beta[i]))
                else:
                    log_crp_prior = np.log(cat_count[cat] / (cluster_count - 1. + self.alpha))
                    log_likelihood = 0
                    for obs in cluster_obs:
                        log_likelihood += np.log((cat_dict[cat].count(obs) + self.beta[i]) / (len(cat_dict[cat]) + self.support_size * self.beta[i]))
                
                log_p_grid[cat_index] = log_crp_prior + log_likelihood
            
            #print(cat_grid, file=sys.stderr)
            #print(lognormalize(log_p_grid), file=sys.stderr)
            #raw_input()
            self.categories[i] = np.random.choice(a = cat_grid, p = lognormalize(log_p_grid))

    def pf_sample_breakpoints(self, trial):
        """Perform sequential sampling on breakpoints.
        """
        if trial == 1: return
        bp_index = trial - 1
        
        # step 2: assign weights to these particles
        for i in xrange(self.sample_size):
            # determine if it is a breakpoint
            left_run, _, last_run_number, _ = self.get_surround_runs(bps = self.breakpoints[i], target_bp = bp_index, data = self.data[:trial])
            if self.prior_type == 'Poisson': 
                l = np.random.poisson(self.l[i])
            elif self.prior_type == 'Geometric':
                l = np.random.geometric(self.l[i])
            
            if len(left_run) == l: 
                endpoints = np.nonzero(self.breakpoints[i][:bp_index])[0]
                numbers = self.breakpoints[i][endpoints]

                # construct the CRP prior
                counts, uniq_numbers, new_number = self.smallest_unused_label(numbers)
                prior_prob = np.bincount(list(numbers) + [new_number])
                prior_grid = range(len(prior_prob)) # this works because bincount orders and fill all missing spots
                
                # set the count of the new category and the last category to 0
                prior_prob[new_number] = 0
                if last_run_number is not None: prior_prob[last_run_number] = 0
                
                # normalize
                N = float(prior_prob.sum())
                prior_prob = prior_prob / (N + self.alpha)
                prior_prob[new_number] = self.alpha / (N + self.alpha)

                # sample the next value
                self.breakpoints[i][bp_index] = np.random.choice(a = prior_grid, p = prior_prob)

            # get statistics after proposal
            total_number_of_runs = np.nonzero(self.breakpoints[i])[0].size
            total_run_length = trial 
            if self.prior_type == 'Poisson':
                self.l[i] = np.random.gamma(shape = self.gamma_prior_shape + total_run_length, 
                                            scale = 1. / (self.gamma_prior_rate + total_number_of_runs))
                if self.l[i] < 1: self.l[i] = 1
            elif self.prior_type == 'Geometric':
                # Beta(alpha + total_number_of_runs, beta + sum(all_run_lengths) - total_number_of_runs)
                self.l[i] = np.random.beta(a = self.geom_prior_alpha + total_number_of_runs, 
                                           b = self.geom_prior_beta + total_run_length)
            # mini gibbs end

        self.reweight(trial = trial)
        self.log_weight = self.log_weight - np.max(self.log_weight)

        # step 3: resample breakpoints if necessary
        weights = lognormalize(self.log_weight)
        ess = 1 / (weights ** 2).sum()
        if ess < self.resample_rate * self.sample_size: 
            top_particle = np.where(weights == weights.max())[0][0]
            top_breakpoints = copy.deepcopy(self.breakpoints[top_particle])
            top_l = copy.deepcopy(self.l[top_particle])

            for i in xrange(self.sample_size):
                for j in xrange(1, trial):
                    top_breakpoints[j] = 0
                    c_number_count, c_uniq_numbers, c_new_number = self.smallest_unused_label(top_breakpoints)
                    grid = np.hstack((c_uniq_numbers, c_new_number))

                    left_run, right_run, left_number, right_number = self.get_surround_runs(bps = top_breakpoints[:trial], target_bp = j, 
                                                                                            data = self.data[:trial])
                    left_len_prior, right_len_prior, both_len_prior = \
                        self.log_length_prior(runs = [left_run, right_run, left_run + right_run], c_l = top_l)

                    grid = np.delete(grid, np.where(grid == left_number)) # handles "None" as well
                    grid = np.delete(grid, np.where(grid == right_number))
                    log_p_grid = np.zeros(len(grid))

                    for number in grid:
                        number_index = np.where(grid == number)[0]
                        top_breakpoints[j] = number
                        # compute the prior
                        if number == 0:
                            log_prior = both_len_prior 
                        else:
                            log_prior = left_len_prior + right_len_prior
                            log_prior += self.log_count_prior2(number = number, number_count = c_number_count, avoid = (left_number, right_number))

                        # compute the log likelihood
                        cats = self.get_categories(bps = top_breakpoints[:trial], data = self.data[:trial])
                        log_likelihood = 0
                        for cat, obs in cats.iteritems():
                            log_likelihood += self.log_joint_prob(obs = obs, beta = self.beta)

                        # compute the log posteior
                        log_p_grid[number_index] = log_prior * self.temp + log_likelihood

                    log_p_grid = log_p_grid - np.max(log_p_grid)
                    top_breakpoints[j] = np.random.choice(a = grid, p = lognormalize(log_p_grid))

                total_number_of_runs = np.nonzero(top_breakpoints)[0].size
                total_run_length = trial 
                if self.prior_type == 'Poisson':
                    top_l = np.random.gamma(shape = self.gamma_prior_shape + total_run_length, 
                                            scale = 1. / (self.gamma_prior_rate + total_number_of_runs))
                    if top_l < 1: top_l = 1
                elif self.prior_type == 'Geometric':
                    # Beta(alpha + total_number_of_runs, beta + sum(all_run_lengths) - total_number_of_runs)
                    top_l = np.random.beta(a = self.geom_prior_alpha + total_number_of_runs, 
                                           b = self.geom_prior_beta + total_run_length)

                self.breakpoints[i] = copy.deepcopy(top_breakpoints)
                self.l[i] = copy.deepcopy(top_l)

            self.log_weight = np.ones(self.sample_size)
            self.log_weight = np.log(self.log_weight / np.sum(self.log_weight))

        return

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

    def log_count_prior2(self, number, number_count, avoid):

        try: target_count = number_count[number]
        except: target_count = self.alpha
        if target_count == 0: target_count = self.alpha
        total_count = number_count[1:].sum() + self.alpha
        for n in np.unique(avoid):
            if n is not None: total_count -= number_count[n]

        return math.log(target_count / total_count)

    def log_joint_prob(self, obs, beta):
        """Calculate the joint probability of all observations of the same category,
        which may contain several runs.
        """
        log_p = math.lgamma(beta * self.support_size) - math.lgamma(beta * self.support_size + len(obs)) 
        for y in self.support:
            log_p += math.lgamma(obs.count(y) + beta) - math.lgamma(beta)
        return log_p

    def reweight(self, trial):

        # The whole thing depends on the factorization
        for i in xrange(self.sample_size):

            # try to get all run lengths - length priors
            #all_runs = self.get_all_runs(bps = self.breakpoints[i][:trial], data = self.data[:trial])
            #log_p = np.sum(self.log_length_prior(runs = all_runs, c_l = self.l[i])) * self.temp
        
            # get all observations indexed by categories 
            log_p = 0
            all_cats = self.get_categories(bps = self.breakpoints[i][:trial], data = self.data[:trial])
        
            # joint probability of the category partition
            #log_p += math.lgamma(self.alpha) + len(all_cats) * math.log(self.alpha) - math.lgamma(self.alpha + len(all_runs))
        
            for cat, obs in all_cats.iteritems():
                number_of_runs_in_cat = np.where(self.breakpoints[i] == cat)[0].size
                log_p += math.lgamma(number_of_runs_in_cat)
                log_p += self.log_joint_prob(obs = obs, beta = self.beta)

           # target_number = self.breakpoints[i][trial - 1]
           # if target_number == 0:
           #     _, _, run_number, _ = self.get_surround_runs(bps = self.breakpoints[i], target_bp = trial-1, data = self.data[:trial])
           # else:
           #     run_number = target_number
                
            #try:
            #    log_p = (all_cats[run_number].count(self.data[trial-1]) + self.beta) / \
            #        (len(all_cats[run_number]) + self.support_size * self.beta)
            #except KeyError:
            #    log_p = 1. / self.support_size

            self.log_weight[i] += log_p

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


    def get_all_runs(self, bps, data=None):

        if data is None: data = self.data
        endpoints = np.nonzero(bps)[0]

        runs = []
        for i in xrange(len(endpoints)):
            if i == len(endpoints) - 1:
                run = data[endpoints[i]:]
            else:
                run = data[endpoints[i]:endpoints[i+1]]
            runs.append(run)
        return runs

    def get_category_flat_dict(self, avoid=None, clusters=None, categories=None, data=None):
        """Returns category-indexed obs.
        """
        if data is None: data = self.data
        if clusters is None: clusters = self.clusters
        if categories is None: categories = self.categories

        cat_flat_dict = {}
        for i in xrange(len(clusters)):
            if avoid is not None and i == avoid: continue
            try: run = data[clusters[i]:clusters[i+1]]
            except IndexError: run = data[clusters[i]:]
            try: cat_flat_dict[categories[i]].extend(run)
            except KeyError: cat_flat_dict[categories[i]] = run
        return cat_flat_dict

    def run(self):
        """Run the sampler.
        """
        if self.s_type == 'batch':
            #header = 'alpha,beta,l,'
            header = 'alpha,l,'
            header += ','.join([str(t) for t in xrange(1, self.total_trial+1)])
            print(header, file = self.sample_output_file)

            for i in xrange(self.sample_size):
                self.iteration = i + 1
                if self.iteration % 50 == 0:
                    print('Iteration:', self.iteration, self.beta, file=sys.stderr)
                    if self.sample_output_file != sys.stdout: self.sample_output_file.flush()

                self.set_temperature()
                self.batch_sample_clusters()
                self.batch_sample_l()
                #self.tversky_sample_l()
                self.batch_sample_categories()
                if self.sample_beta: self.batch_sample_beta()
                #print('l:', self.l, file=sys.stderr)
                #print('clusters:', self.clusters, file=sys.stderr)
                #raw_input()
                #print('beta:', self.beta, file=sys.stderr)
                #print('categories:', self.categories, file=sys.stderr)
                #raw_input()
                self.print_batch_iteration(dest = self.sample_output_file)

        elif self.s_type == 'increm':
            # currently we output samples and predictions at the same time
            headers = 'trial.no,particle.no,weight,l,'
            headers += ','.join(['bp.' + str(t) for t in xrange(1, self.total_trial)])
            print(headers, file=self.sample_output_file)
            
            self.increm_predict_next_trial(next_bp = 0)
            
            for i in xrange(self.total_trial):
                self.iteration = i+1
                self.set_temperature()
                self.pf_sample_breakpoints(trial = self.iteration)

                try: self.increm_predict_next_trial(next_bp = self.iteration)
                except IndexError: pass

                weights = lognormalize(self.log_weight)
                # debug info
                if i % 20 == 0: print('Trial:', self.iteration, np.dot(weights, self.l), file=sys.stderr)
                if i % 100 == 0: self.sample_output_file.flush()

                self.print_increm_iteration(dest = self.sample_output_file)

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
        for i in xrange(len(self.clusters)):
            try: cat_seq.extend([self.categories[i]] * (self.clusters[i+1] - self.clusters[i]))
            except IndexError: cat_seq.extend([self.categories[i]] * (self.total_trial - self.clusters[i]))
        #print(cat_seq)
        output += ','.join([str(c) for c in self.reorder_labels(cat_seq)])
        print(output, file = dest)

    def print_increm_iteration(self, dest):
        
        weights = lognormalize(self.log_weight)
        
        for p in xrange(self.sample_size):
            print(self.iteration, p, weights[p].round(decimals = 5),
                  self.l[p].round(decimals = 5), *self.reorder_labels(self.breakpoints[p][:self.iteration]), 
                  sep=',', file=self.sample_output_file)

    def reorder_labels(self, labels):
        labels = np.array(labels)
        cur_labels = uniqify(labels[np.where(labels > 0)])
        new_labels = range(1,len(cur_labels) + 1)
        labels_copy = copy.deepcopy(labels)
        for i in xrange(len(cur_labels)):
            labels_copy[np.where(labels == cur_labels[i])] = new_labels[i]
        return labels_copy
