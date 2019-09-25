import numpy as np
import copy
from tqdm.auto import tqdm
import seaborn as sns


class SearchMixin:

    def __init__(self, model, grid_params=None, random_params=None, n_random=60, metric=None, random_state=None):
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
        if bool(self.grid_params) and bool(self.random_params):
            return self._generate_mixed_grid(self.grid_params, self.random_params, self.n_random, random_state=random_state)
        elif bool(self.grid_params) and bool(not self.random_params):
            return self._generate_grid_grid(self.grid_params)
        elif bool(self.random_params) and bool(not self.grid_params):
            return self._generate_random_grid(self.random_params, self.n_random, random_state=random_state)
        else:
            raise ValueError("Need at least one of random parameters or grid parameters!")

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

    def _generate_random_grid(self, params, n_random, random_state=None):
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
        random_list = np.random.randint(0, 36000, size=n_random)
        param_list = []
        for index in range(n_random):
            param_list.append({key: params[key].rvs(1, random_state=random_list[index])[0] for key in params})
        return param_list

    def _generate_grid_grid(self, params):
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

    def _generate_mixed_grid(self, grid_params, random_params, n_random, random_state=None):
        '''
        Combines both grid and random hyperparameter generation together.  Input parameters are the same as above.
        '''
        from itertools import product
        grid = generate_grid_grid(grid_params)
        random = generate_random_grid(random_params, n_random, random_state)
        return list(product(*[grid, random]))


class ResampleSearch(SearchMixin):

    def __init__(self, model, grid_params=None, random_params=None, num_samples=10, sample_size=0.2, test_size=0.3,
                 n_random=60, metric=None, random_state=None):
        '''
        Hyperparameter tuning using resampling to estimate accuracy

        :param model: model to tune
        :param grid_params: Parameters to grid search over
        :param random_params: Parameters to random search over
        :param num_samples: number of samples to train model on
        :param sample_size: Float representing size of samples relative to full dataset
        :param test_size: Float representing the test size in each sample
        :param n_random: Number of random parameters to use in search
        :param metric: scoring metric
        :param random_state: duh
        '''
        from sklearn.model_selection import train_test_split
        self.train_test_split = train_test_split
        self.num_samples = num_samples
        self.sample_size = sample_size
        self.test_size = test_size
        super().__init__(model=model, grid_params=grid_params, random_params=random_params, n_random=n_random,
                         metric=metric, random_state=random_state)


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


class CrossvalSearch(SearchMixin):

    def __init__(self, model, grid_params=None, random_params=None, cv=5, stratified=True, shuffle=True,
                 n_random=60, metric=None, random_state=None):
        '''
        Hyperparameter search using k-fold crossvalidation to estimate accuracy

        :param model: model to tune
        :param grid_params: Parameters to grid search over
        :param random_params: Parameters to random search over
        :param cv: number of crossvalidation folds
        :param stratified: True to stratify crossvalidation folds
        :param shuffle: True to shuffle prior to splitting
        :param n_random: Number of random parameters to use in search
        :param metric: metric to use as score
        :param random_state: not sure what this does
        '''
        self.stratified = stratified
        self.cv = cv
        self.shuffle = shuffle
        super().__init__(model=model, grid_params=grid_params, random_params=random_params, n_random=n_random,
                         metric=metric, random_state=random_state)

    def _kfolds(self, X, y):
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
        for train_ndx, test_ndx in tqdm(self._kfolds(X, y), disable=(not verbose)):
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


class SimpleSearch(SearchMixin):

    def __init__(self, model, grid_params=None, random_params=None, val_set=None, n_random=60, metric=None, random_state=None):
        '''
        Hyperparameter search using a simple train/test split to estimate accuracy

        :param model: model to tune
        :param grid_params: Parameters to grid search over
        :param random_params: Parameters to random search over
        :param val_set: validation set to compare model accuracy to.  If None, uses accuracy on training set. If a tuple,
            uses the first component of tuple as validation features and the second component as validation labels.  If
            a float, splits the set into train/validation sets with validation set size being determined by val_set (ie,
            if val_set=0.3, train set is 70% of the input data and the validation set is 30%)
        :param n_random: Number of random parameters to use in search
        :param metric: metric to score
        :param random_state: duh
        '''
        self.val_set = val_set
        if type(val_set)==tuple:
            self.wants_val = True
            self.x_val = val_set[0]
            self.y_val = val_set[1]
        elif type(val_set)==float:
            from sklearn.model_selection import train_test_split
            self.split = train_test_split
            self.wants_val = True
        else: self.wants_val = False
        super().__init__(model=model, grid_params=grid_params, random_params=random_params, n_random=n_random,
                         metric=metric, random_state=random_state)

    def _eval(self, model, X, y, verbose=False, random_state=None):
        if self.wants_val:
            if type(self.val_set)==float:
                X, x_val, y, y_val = self.split(X, y, test_size=self.val_set, stratify=y, random_state=random_state)
            else:
                x_val = self.x_val
                y_val = self.y_val
        else:
            x_val = X
            y_val = y
        model.fit(X, y)
        score = self.metric(y_val, model.predict(x_val))
        return score, [score]


