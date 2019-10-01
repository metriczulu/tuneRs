import numpy as np

class Uniform:

    def __init__(self, lower, upper, bins=None, dtype="int"):
        '''
        Used for uniform distributions

        :param lower: Lower bound of distribution
        :param upper: Upper bound of distribution
        :param bins: Bins for numbers.  Each bin has equal chance of being pulled.  If None, no bins are used
        :param dtype: Dtype for distribution.  "int", "float", and "float32" are all valid inputs.
        '''
        self.lower = lower
        self.upper = upper
        self.dtype = dtype
        self.bins = bins

    def _single_rvs(self, lower, upper, random_state=None):
        if random_state is None:
            random_state = np.random.randint(0, 36e7)
        np.random.seed(random_state)
        if self.dtype == "int":
            return np.random.randint(lower, upper+1)
        elif self.dtype == "float":
            return np.random.uniform(lower, upper)
        elif self.dtype == "float32":
            return np.random.uniform(lower, upper).astype(np.float32)

    def rvs(self, num_samples=1, random_state=None):
        '''
        Samples the distribution

        :param num_samples: Number of elements in the sample
        :param random_state: Random state
        :return: List of samples
        '''
        if random_state is None:
            random_state = np.random.randint(0, 36e7)
        np.random.seed(random_state)
        if self.bins is None:
            if self.dtype=="int":
                return np.random.randint(self.lower, self.upper+1, num_samples)
            elif self.dtype=="float":
                return np.random.uniform(self.lower, self.upper, num_samples)
            elif self.dtype=="float32":
                return np.random.uniform(self.lower, self.upper, num_samples).astype(np.float32)
        else:
            random_list = []
            num_bins = len(self.bins)
            rand_list = np.random.randint(0, 36e7, num_samples)
            for num in range(num_samples):
                bin_range = self.bins[np.random.randint(0, num_bins)]
                random_list.append(self._single_rvs(bin_range[0], bin_range[1], random_state=rand_list[num]))
            return random_list

class Normal:

    def __init__(self, mean, sd, min=None, max=None, dtype="float"):
        '''
        Normal distribution

        :param mean: Mean of distribution
        :param sd: Standard deviation of distribution
        :param max: Maximum allowable sample.  If None, no max is considered
        :param min: Minimum allowable sample.  If None, no min is considered
        :param dtype: Dtype for distribution.  "int", "float", and "float32" are all valid inputs.
        '''
        self.mean = mean
        self.sd = sd
        self.dtype = dtype
        self.max = max
        self.min = min
        self.lower = max # added for compatibility with Bayesian search methods
        self.upper = min # added for compatibility with Bayesian search methods

    def _single_rvs(self, random_state=None):
        if random_state is None:
            random_state = np.random.randint(0, 36e7)
        np.random.seed(random_state)
        if self.dtype == "int":
            num = int(np.random.normal(self.mean, self.sd))
        elif self.dtype == "float":
            num = np.random.normal(self.mean, self.sd)
        elif self.dtype == "float32":
            num = np.float32(np.random.normal(self.mean, self.sd))
        if self.min and self.max:
            if (self.min <= num) and (num <= self.max): return num
            else: return self._single_rvs(random_state=random_state + 1)
        elif self.min and not self.max:
            if self.min <= num: return num
            else: return self._single_rvs(random_state=random_state + 1)
        elif self.max and not self.min:
            if num <= self.max: return num
            else: return self._single_rvs(random_state=random_state + 1)

    def rvs(self, num_samples=1, random_state=None):
        if random_state is None:
            random_state = np.random.randint(0, 36e7)
        np.random.seed(random_state)
        if (self.max is None) and (self.min is None):
            if self.dtype=="int":
                return int(np.random.normal(self.mean, self.sd, num_samples))
            elif self.dtype=="float":
                return np.random.normal(self.mean, self.sd, num_samples)
            elif self.dtype=="float32":
                return np.float32(np.random.normal(self.mean, self.sd, num_samples))
        else:
            random_list = []
            rand_list = np.random.randint(0, 36e7, num_samples)
            for index in range(num_samples):
                random_list.append(self._single_rvs(random_state=rand_list[index]))
            return random_list

class LogUniform:

    def __init__(self, lower, upper, granularity=1000, replace=True, reverse=False, dtype="float"):
        '''
        Log normal distribution

        :param lower: Smallest number in distribution
        :param upper: Largest number in distribution
        :param granularity: Higher numbers can produce more diverse samples
        :param replace: True to allow the same number to be sampled multiple times
        :param reverse: True to have the distribution go from upper to lower instead
        :param dtype: Dtype for distribution.  "int", "float", and "float32" are all valid inputs.
        '''
        self.lower = lower
        self.upper = upper
        self.dtype = dtype
        self.granularity = granularity
        self.replace = replace
        self.reverse = reverse

    def rvs(self, num_samples=1, random_state=None):
        if random_state is None:
            random_state = np.random.randint(0, 36e7)
        np.random.seed(random_state)
        if self.dtype == "int":
            dist = np.random.choice(np.geomspace(self.lower, self.upper, num_samples*self.granularity, dtype=int),
                                          num_samples, replace=self.replace)
        elif self.dtype == "float":
            dist = np.random.choice(np.geomspace(self.lower, self.upper, num_samples*self.granularity), num_samples,
                                         replace=self.replace)
        elif self.dtype == "float32":
            dist = np.random.choice(np.geomspace(self.lower, self.upper, num_samples*self.granularity,
                                                            dtype=np.float32), num_samples, replace=self.replace)
        if self.reverse:
            dist = [(self.upper - num) for num in dist]
        return dist

class Categorical:

    def __init__(self, categories, probs=None, replace=True):
        '''
        Distribution for categorical variables

        :param categories: List of categorical variables
        :param probs: list of probabilities for each category.  If None, all categories have equal cance of being sampled
        :param replace: True to allow the same category to be sampled multiple times
        '''
        self.categories = categories
        self.cat_index = list(range(len(categories)))
        if probs:
            self.default_probs = False
            self.probs = probs/np.sum(probs)
        else:
            self.default_probs = True
            self.probs = [1/len(categories)]*len(categories)
        self.replace = replace

    def rvs(self, num_samples=1, random_state=None):
        if random_state is None:
            random_state = np.random.randint(0, 36e7)
        np.random.seed(random_state)
        index_dist = np.random.choice(self.cat_index, num_samples, replace=self.replace, p=self.probs)
        return [self.categories[index] for index in index_dist]

    def __add__(self, other_cat):
        if self.default_probs and other_cat.default_probs:
            return Categorical(self.categories+other_cat.categories, replace=(self.replace and other_cat.replace))
        else:
            return Categorical(self.categories+other_cat.categories, self.probs+other_cat.probs,
                               replace=(self.replace and other_cat.replace))

class Concatenate:

    def __init__(self, categories, probs=None):
        '''
        Concatenate two distributions together

        :param categories: List of distributions
        :param probs: list of probabilities for each distribution.  If None, all categories have equal chance of
            being sampled
        '''
        self.categories = categories
        if probs:
            self.default_probs = False
            self.probs = probs/np.sum(probs)
        else:
            self.default_probs = True
            self.probs = [1/len(dist_list)]*len(dist_list)

    def rvs(self, num_samples=1, random_state=None):
        if random_state is None:
            random_state = np.random.randint(0, 36e7)
        np.random.seed(random_state)
        rand_list = np.random.randint(0, 36e7, num_samples)
        samples = np.random.choice(self.categories, num_samples, replace=True, p=self.probs)
        sampled_samples = []
        for index, sample in enumerate(samples):
            sampled_samples.append(sample.rvs(1, random_state=rand_list[index])[0])
        return sampled_samples

    def __add__(self, other_dist):
        if self.default_probs and other_dist.default_probs:
            return Concatenate(self.categories+other_dist.categories)
        else:
            return Concatenate(self.categories+other_dist.categories, self.probs+other_dist.probs)

    def adjust_probs(self, probs=None):
        if probs:
            self.default_probs = False
            self.probs = probs/np.sum(probs)
        else:
            self.default_probs = True
            self.probs = [1/len(dist_list)]*len(dist_list)

class Cartesian:

    def __init__(self, dist_list, condition_function=None, type="tuple"):
        '''
        Draw random samples from the cartesian product space of multiple space distributions

        :param dist_list: List of distributions to draw from
        :param type: If "tuple", return a tuple.  Else, return a list
        :param condition_function: If None, draw randomly from entire cartesian product space.  Else, input a function
            which maps any element (a, b, c,..., n) in the Cartestian production space AxBxCx...xN to either True or False.
            Only sample elements which for which condition_function is True.
        '''
        self.dist_list = dist_list
        self.type = type
        self.size = len(self.dist_list)
        if condition_function is None:
            self.condition_function = lambda x: True
        else:
            self.condition_function = condition_function

    def _single_rvs(self, random_state=None):
        if random_state is None:
            random_state = np.random.randint(0, 36e7)
        np.random.seed(random_state)
        draw_list = []
        rand_list = np.random.randint(0, 36e7, self.size)
        for index, dist in enumerate(self.dist_list):
            draw = dist.rvs(1, random_state=rand_list[index])[0]
            draw_list.append(draw)
        if self.type=="tuple":
            draw_list = tuple(draw_list)
        if not self.condition_function(draw_list):
            draw_list = self._single_rvs(random_state=random_state+42)
        return draw_list


    def rvs(self, num_samples, random_state=None):
        if random_state is None:
            random_state = np.random.randint(0, 36e7)
        np.random.seed(random_state)
        rand_list = np.random.randint(0, 36e7, num_samples)
        rvs_list = []
        for sample in range(num_samples):
            rvs_list.append(self._single_rvs(random_state=rand_list[sample]))
        return rvs_list
