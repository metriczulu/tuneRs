
class Uniform:

    def __init__(self, lower, upper, bins=None, dtype="int"):
        '''
        Used for uniform distributions

        :param lower: Lower bound of distribution
        :param upper: Upper bound of distribution
        :param bins: Bins for numbers.  Each bin has equal chance of being pulled.  If None, no bins are used
        :param dtype: Dtype for distribution.  "int", "float", and "float32" are all valid inputs.
        '''
        import numpy as np
        self.np = np
        self.lower = lower
        self.upper = upper
        self.dtype = dtype
        self.bins = bins

    def _single_rvs(self, lower, upper, random_state=None):
        if random_state is None:
            random_state = self.np.random.randint(0, 36e7)
        self.np.random.seed(random_state)
        if self.dtype == "int":
            return self.np.random.randint(lower, upper+1)
        elif self.dtype == "float":
            return self.np.random.uniform(lower, upper)
        elif self.dtype == "float32":
            return self.np.random.uniform(lower, upper).astype(self.np.float32)

    def rvs(self, num_samples, random_state=None):
        '''
        Samples the distribution

        :param num_samples: Number of elements in the sample
        :param random_state: Random state
        :return: List of samples
        '''
        if random_state is None:
            random_state = self.np.random.randint(0, 36e7)
        self.np.random.seed(random_state)
        if self.bins is None:
            if self.dtype=="int":
                return self.np.random.randint(self.lower, self.upper+1, num_samples)
            elif self.dtype=="float":
                return self.np.random.uniform(self.lower, self.upper, num_samples)
            elif self.dtype=="float32":
                return self.np.random.uniform(self.lower, self.upper, num_samples).astype(self.np.float32)
        else:
            random_list = []
            num_bins = len(self.bins)
            rand_list = self.np.random.randint(0, 36e7, num_samples)
            for num in range(num_samples):
                bin_range = self.bins[self.np.random.randint(0, num_bins)]
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
        import numpy as np
        self.np = np
        self.mean = mean
        self.sd = sd
        self.dtype = dtype
        self.max = max
        self.min = min

    def _single_rvs(self, random_state=None):
        if random_state is None:
            random_state = self.np.random.randint(0, 36e7)
        self.np.random.seed(random_state)
        if self.dtype == "int":
            num = int(self.np.random.normal(self.mean, self.sd))
        elif self.dtype == "float":
            num = self.np.random.normal(self.mean, self.sd)
        elif self.dtype == "float32":
            num = self.np.float32(self.np.random.normal(self.mean, self.sd))
        if self.min and self.max:
            if (self.min <= num) and (num <= self.max): return num
            else: return self._single_rvs(random_state=random_state + 1)
        elif self.min and not self.max:
            if self.min <= num: return num
            else: return self._single_rvs(random_state=random_state + 1)
        elif self.max and not self.min:
            if num <= self.max: return num
            else: return self._single_rvs(random_state=random_state + 1)

    def rvs(self, num_samples, random_state=None):
        if random_state is None:
            random_state = self.np.random.randint(0, 36e7)
        self.np.random.seed(random_state)
        if (self.max is None) and (self.min is None):
            if self.dtype=="int":
                return int(self.np.random.normal(self.mean, self.sd, num_samples))
            elif self.dtype=="float":
                return self.np.random.normal(self.mean, self.sd, num_samples)
            elif self.dtype=="float32":
                return np.float32(self.np.random.normal(self.mean, self.sd, num_samples))
        else:
            random_list = []
            rand_list = self.np.random.randint(0, 36e7, num_samples)
            for index in range(num_samples):
                random_list.append(self._single_rvs(random_state=rand_list[index]))
            return random_list

class LogNormal:

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
        import numpy as np
        self.np = np
        self.lower = lower
        self.upper = upper
        self.dtype = dtype
        self.granularity = granularity
        self.replace = replace
        self.reverse = reverse

    def rvs(self, num_samples, random_state=None):
        if random_state is None:
            random_state = self.np.random.randint(0, 36e7)
        self.np.random.seed(random_state)
        if self.dtype == "int":
            dist = self.np.random.choice(self.np.geomspace(self.lower, self.upper, num_samples*self.granularity, dtype=int),
                                          num_samples, replace=self.replace)
        elif self.dtype == "float":
            dist = self.np.random.choice(self.np.geomspace(self.lower, self.upper, num_samples*self.granularity), num_samples,
                                         replace=self.replace)
        elif self.dtype == "float32":
            dist = self.np.random.choice(self.np.geomspace(self.lower, self.upper, num_samples*self.granularity,
                                                            dtype=self.np.float32), num_samples, replace=self.replace)
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
        import numpy as np
        self.np = np
        self.categories = categories
        self.cat_index = list(range(len(categories)))
        if probs:
            self.probs = probs/np.sum(probs)
        else:
            self.probs = [1/len(categories)]*len(categories)
        self.replace = replace

    def rvs(self, num_samples, random_state=None):
        if random_state is None:
            random_state = self.np.random.randint(0, 36e7)
        self.np.random.seed(random_state)
        index_dist = self.np.random.choice(self.cat_index, num_samples, replace=self.replace, p=self.probs)
        return [self.categories[index] for index in index_dist]

    def __add__(self, other_cat):
        return Categorical(self.categories+other_cat.categories, self.probs+other_cat.probs, replace = (self.replace and other_cat.replace))

class Concatenate:

    def __init__(self, categories, probs=None):
        '''
        Concatenate two distributions together

        :param categories: List of distributions
        :param probs: list of probabilities for each distribution.  If None, all categories have equal chance of being sampled
        '''
        import numpy as np
        self.np = np
        self.categories = categories
        if probs:
            self.probs = probs/np.sum(probs)
        else:
            self.probs = [1/len(dist_list)]*len(dist_list)

    def rvs(self, num_samples, random_state=None):
        if random_state is None:
            random_state = self.np.random.randint(0, 36e7)
        self.np.random.seed(random_state)
        rand_list = self.np.random.randint(0, 36e7, num_samples)
        samples = self.np.random.choice(self.categories, num_samples, replace=True, p=self.probs)
        sampled_samples = []
        for index, sample in enumerate(samples):
            sampled_samples.append(sample.rvs(1, random_state=rand_list[index])[0])
        return sampled_samples

    def __add__(self, other_dist):
        return Concatenate(self.categories+other_dist.categories, self.probs+other_dist.probs)

    def adjust_probs(self, probs=None):
        if probs:
            self.probs = probs/self.np.sum(probs)
        else:
            self.probs = [1/len(dist_list)]*len(dist_list)
