import numpy as np
import copy
from tqdm.auto import tqdm
import seaborn as sns

def generate_random_grid(params, num_iter, random_state=None):
    '''
    Generates a list of random hyperparameter combinations

    :param params: Dictionary of possible param values.  Can accept skopt.space and tuneRs.space objects
    :param num_iter: Number of random hyperparameter combinations to pull
    :param random_state: Random state
    :return: returns a list of hyperparameter dictionaries
    '''
    if random_state is None:
        random_state = np.random.randint(0, 36000)
    np.random.seed(random_state)
    random_list = np.random.randint(0, 36000, size=num_iter)
    param_list = []
    for index in range(num_iter):
        param_list.append({key: params[key].rvs(1, random_state=random_list[index])[0] for key in params})
    return param_list


def generate_grid_grid(params):
    '''
    Generates a list of all hyperparameter combinations
    :param params: Dictionary of param values.  Uses lists.
    :return: returns a list of hyperparameter dictionaries
    '''
    from itertools import product
    key_list = []
    param_list = []
    for key in params:
        key_list.append(key)
        param_list.append(params[key])
    cartesian_product = product(*param_list)
    param_dict_list = []
    param_size = len(key_list)
    for param in cartesian_product:
        temp_dict = {key_list[i]: param[i] for i in range(param_size)}
        param_dict_list.append(temp_dict)
    return param_dict_list


def generate_mixed_grid(grid_params, random_params, num_random, random_state=None):
    '''
    Combines both grid and random hyperparameter generation together.  Input parameters are the same as above.
    '''
    from itertools import product
    grid = generate_grid_grid(grid_params)
    random = generate_random_grid(random_params, num_random, random_state)
    return list(product(*[grid, random]))


class SearchMixin:

    def __init__(self, model, grid_params, random_params, n_random=60, metric=None, random_state=None):
        '''
        Mixin class for all searches.

        :param model: model to tune.  Must have .fit() and .predict() methods in the scikit-learn style
        :param params: parameter dictionary
        :param metric: performance metric
        :param random_state: random state
        '''
        self._set_params(model, grid_params, random_params, n_random, metric, random_state)

    def _set_params(self, model, grid_params, random_params, n_random, metric=None, random_state=None):
        '''
        Sets all initial params
        '''
        if metric is None:
            from sklearn.metrics import accuracy_score
            self.metric = accuracy_score
        else:
            self.metric = metric
        self.model = model
        if random_state is None:
            random_state = np.random.randint(0, 36e6)
        self.best_params_ = None
        self.best_distribution_ = []
        self.random_state = random_state
        self.best_score_ = 0.0
        self.best_estimator_ = None
        self.grid_params = grid_params
        self.random_params = random_params
        self.n_random = n_random
        self.param_grid = self._generate_grid()

    def _generate_grid(self, random_state=None):
        #return parameter grid dictionary
        return dict()

    def _eval(self, model, X, y, random_state=None, verbose=False):
        #return score, distribution
        return float, list

    def _fit(self, X, y, train_best_estimator=True, verbose=False, super_verbose=False):
        '''
        Trains every hyperparameter combination to find the best

        :param X: Features
        :param y: Labels
        :param train_best_estimator: If True, train the best model on all data
        :param verbose: True to generate progress bar that tracks count of parameter combinations
        :param super_verbose: True to use verbose=True for each call of _eval as well
        :return: returns the estimator with best parameters
        '''
        np.random.seed(self.random_state)
        random_list = np.random.randint(0, 36e6, size=len(self.param_grid))
        for index, param in tqdm(enumerate(self.param_grid), disable=(not verbose)):
            self.model.set_params(**param)
            temp_model = copy.deepcopy(self.model)
            temp_model.set_params(**param)
            score, distribution = self._eval(temp_model, X, y, random_state=random_list[index], verbose=super_verbose)
            if score > self.best_score_:
                self.best_params_ = param
                self.best_score_ = score
                self.best_distribution_ = distribution
        self.best_estimator_ = self.model.set_params(**self.best_params_)
        if train_best_estimator:
            self.best_estimator_.fit(X, y)
        return self.best_estimator_

    def fit(self, X, y, train_best_estimator=True, verbose=False, super_verbose=False):
        #
        # self.fit() and self._fit() are separate methods so that in Bayesian search classes self._fit() can be
        # called multiple times with different parameter grids
        #
        return self._fit(X, y, train_best_estimator=train_best_estimator, verbose=verbose, super_verbose=super_verbose)

    def plot_best(self, color="orange", linecolor="orangered", figsize=(12, 8)):
        '''
        Plots the distribution of scores for the best parameters found

        :param color: Color of histogram
        :param linecolor: Color of line
        :param figsize: Figure size
        '''
        plt.figure(figsize=figsize)
        plt.title("Sample Accuracy Distribution")
        plt.xlabel("Observed Accuracy")
        sns.distplot(self.best_distribution_, color=color, kde_kws={'color': linecolor, 'linewidth': 3})


class RSMixin(SearchMixin):

    def __init__(self, model, params, num_samples=10, sample_size=0.2, test_size=0.3, metric=None, random_state=None):
        '''
        Mixin class for resample searches

        :param model: model to tune
        :param params: parameter grid to search
        :param num_samples: number of samples to train model on
        :param sample_size: Float representing size of samples relative to full dataset
        :param test_size: Float representing the test size in each sample
        :param metric: scoring metric
        :param random_state: duh
        '''
        from sklearn.model_selection import train_test_split
        self.train_test_split = train_test_split
        self.num_samples = num_samples
        self.sample_size = sample_size
        self.test_size = test_size
        super().__init__(model, params, metric, random_state)

    def _eval(self, model, X, y, random_state=None, verbose=False):
        if random_state is None:
            random_state = np.random.randint(0, 36e6)
        np.random.seed(random_state)
        random_list = np.random.randint(0, 36e6, size=self.num_samples)
        sample_scores = []
        for sample_ndx in tqdm(range(self.num_samples), disable=(not verbose)):
            X_sample, _, y_sample, _ = self.train_test_split(X, y, train_size=self.sample_size, stratify=y,
                                                        random_state=random_list[sample_ndx] + 17)
            X_train, X_test, y_train, y_test = self.train_test_split(X_sample, y_sample, test_size=self.test_size,
                                                                stratify=y_sample, random_state=random_list[sample_ndx])
            clf = model.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            sample_scores.append(self.metric(y_test, y_pred))
            mean = np.mean(sample_scores)
        return mean, sample_scores


class RandomSearchResample(RSMixin):

    def __init__(self, model, params, n_iter=60, num_samples=10, sample_size=0.2, test_size=0.3, metric=None,
                 random_state=None):
        '''
        Random hyperparameter search with resampling (RSMixin)
        '''
        self.n_iter = n_iter
        super().__init__(model, params, num_samples, sample_size, test_size, metric, random_state)
        self.param_grid = self._generate_grid()

    def _generate_grid(self):
        return generate_random_grid(self.params, self.n_iter, self.random_state*42)


class GridSearchResample(RSMixin):

    def __init__(self, model, params, num_samples=10, sample_size=0.2, test_size=0.3, metric=None,
                 random_state=None):
        '''
        Grid search with resampling
        '''
        super().__init__(model, params, num_samples, sample_size, test_size, metric, random_state)
        self.param_grid = self._generate_grid()

    def _generate_grid(self):
        return generate_grid_grid(self.params)


class MixedSearchResample(SearchMixin):

    def __init__(self, model, grid_params, random_params, n_random=5, num_samples=10, sample_size=0.2, test_size=0.3, metric=None,
                 random_state=None):
        self.grid_params = grid_params
        self.random_params = random_params
        self.n_random = n_random
        super().__init__(model, grid_params, num_samples, sample_size, test_size, metric, random_state)
        self.param_grid = self._generate_grid()

    def _generate_grid(self):
        return generate_mixed_grid(self.grid_params, self.random_params, self.n_random, random_state=self.random)


class CVSearchMixin(SearchMixin):

    def __init__(self, model, params, cv=5, stratified=True, shuffle=True, metric=None, random_state=None):
        '''
        Mixin class for k-fold crossvalidation searches

        :param model: model to tune
        :param params: hyperparameters to search
        :param cv: number of crossvalidation folds
        :param stratified: True to stratify crossvalidation folds
        :param shuffle: True to shuffle prior to splitting
        :param metric: metric to use as score
        :param random_state: not sure what this does
        '''
        self.stratified = stratified
        self.cv = cv
        self.shuffle = shuffle
        super().__init__(model, params, metric, random_state)

    def kfolds(self, X, y):
        if self.stratified:
            from sklearn.model_selection import StratifiedKFold
            return StratifiedKFold(n_splits=self.cv, shuffle=self.shuffle, random_state=self.random_state).split(X, y)
        else:
            from sklearn.model_selection import KFold
            return KFold(n_splits=self.cv, shuffle=self.shuffle, random_state=self.random_state).split(X, y)

    def _eval(self, model, X, y, random_state=None, verbose=False):
        if random_state is None:
            random_state = np.random.randint(0, 36e6)
        np.random.seed(random_state)
        sample_scores = []
        for train_ndx, test_ndx in tqdm(self.kfolds(X, y), disable=(not verbose)):
            try:
                x_train, x_test = X.iloc[train_ndx], X.iloc[test_ndx]
                y_train, y_test = y.iloc[train_ndx], y.iloc[test_ndx]
            except:
                x_train, x_test = X[train_ndx], X[test_ndx]
                y_train, y_test = y[train_ndx], y[test_ndx]
            clf = model.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            sample_scores.append(self.metric(y_test, y_pred))
        mean = np.mean(sample_scores)
        return mean, sample_scores


class RandomSearchCrossval(CVSearchMixin):

    def __init__(self, model, params, n_iter=60, cv=5, stratified=True, shuffle=True, metric=None,
                 random_state=None):
        '''
        Random parameter search with crossvalidation scoring
        '''
        self.n_iter = n_iter
        super().__init__(model, params, cv, stratified, shuffle, metric, random_state)
        self.param_grid = self._generate_grid()

    def _generate_grid(self):
        return generate_random_grid(self.params, self.n_iter, self.random_state * 42)


class GridSearchCrossval(CVSearchMixin):
    def __init__(self, model, params, cv=5, stratified=True, shuffle=True, metric=None,
                 random_state=None):
        '''
        Grid parameter search with crossvalidation scoring
        '''
        super().__init__(model, params, cv, stratified, shuffle, metric, random_state)
        self.param_grid = self._generate_grid()

    def _generate_grid(self):
        return generate_grid_grid(self.params)


class SimpleSearchMixin(SearchMixin):

    def __init__(self, model, params, val_set, metric=None, random_state=None):
        '''
        Mixin class for simple comparison tuning

        :param model: model to tune
        :param params: parameters to search
        :param val_set: validation set to compare model accuracy to
        :param metric: metric to score
        :param random_state: duh
        '''
        self.x_val = val_set[0]
        self.y_val = val_set[1]
        super().__init__(model, params, metric, random_state)

    def _eval(self, model, X, y, verbose=False, random_state=None):
        model.fit(X, y)
        score = self.metric(self.y_val, model.predict(self.x_val))
        return score, [score]


class RandomSearchSimple(SimpleSearchMixin):

    def __init__(self, model, params, val_set, n_iter=60, metric=None, random_state=None):
        '''
        Random parameter search with simple scoring
        '''
        self.n_iter = n_iter
        super().__init__(model, params, val_set, metric, random_state)
        self.param_grid = self._generate_grid()

    def _generate_grid(self):
        return generate_random_grid(self.params, self.n_iter, self.random_state * 42)


class GridSearchSimple(SimpleSearchMixin):

    def __init__(self, model, params, val_set, metric=None, random_state=None):
        '''
        Grid parameter search with simple scoring.
        '''
        super().__init__(model, params, val_set, metric, random_state)
        self.param_grid = self._generate_grid()

    def _generate_grid(self):
        return generate_grid_grid(self.params)