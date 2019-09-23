def generate_random_grid(params, num_iter, random_state=None):
    import numpy as np
    if random_state is None:
        random_state = np.random.randint(0, 36000)
    np.random.seed(random_state)
    random_list = np.random.randint(0, 36000, size=num_iter)
    param_list = []
    for index in range(num_iter):
        param_list.append({key: params[key].rvs(1, random_state=random_list[index])[0] for key in params})
    return param_list


def generate_grid_grid(params):

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
    from itertools import product
    grid = generate_grid_grid(grid_params)
    random = generate_random_grid(random_params, num_random, random_state)
    return list(product(*[grid, random]))

class SearchMixin:

    def __init__(self, model, params, metric=None, random_state=None):
        '''

        :param model: model to tune.  Must have .fit() and .predict() methods in the scikit-learn style
        :param params:
        :param metric:
        :param random_state:
        '''
        import numpy as np
        import copy
        from tqdm.auto import tqdm
        self.tqdm = tqdm
        self.copy = copy
        self.np = np
        self._set_params(model, params, metric, random_state)

    def _set_params(self, model, params, metric=None, random_state=None):
        if metric is None:
            from sklearn.metrics import accuracy_score
            self.metric = accuracy_score
        else:
            self.metric = metric
        self.model = model
        if random_state is None:
            random_state = self.np.random.randint(0, 36e6)
        self.best_params_ = None
        self.best_distribution_ = []
        self.random_state = random_state
        self.best_score_ = 0.0
        self.best_estimator_ = None
        self.params = params
        self.param_grid = self._generate_grid()

    def _generate_grid(self, random_state=None):
        #return parameter grid dictionary
        return dict()

    def _eval(self, model, X, y, random_state=None, verbose=False):
        #return score, distribution
        return float, list

    def fit(self, X, y, train_best_estimator=True, verbose=False, super_verbose=False):
        self.np.random.seed(self.random_state)
        random_list = self.np.random.randint(0, 36e6, size=len(self.param_grid))
        for index, param in self.tqdm(enumerate(self.param_grid), disable=(not verbose)):
            self.model.set_params(**param)
            temp_model = self.copy.deepcopy(self.model)
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

    def plot_best(self, color="orange", linecolor="orangered", figsize=(12, 8)):
        try:
            import seaborn as sns
        except:
            raise ValueError("Seaborn is needed to plot the distribution.")
        plt.figure(figsize=figsize)
        plt.title("Sample Accuracy Distribution")
        plt.xlabel("Observed Accuracy")
        sns.distplot(self.best_distribution_, color=color, kde_kws={'color': linecolor, 'linewidth': 3})

class RSMixin(SearchMixin):

    def __init__(self, model, params, num_samples=10, sample_size=0.2, test_size=0.3, metric=None, random_state=None):
        from sklearn.model_selection import train_test_split
        self.train_test_split = train_test_split
        self.num_samples = num_samples
        self.sample_size = sample_size
        self.test_size = test_size
        super().__init__(model, params, metric, random_state)

    def _eval(self, model, X, y, random_state=None, verbose=False):
        if random_state is None:
            random_state = self.np.random.randint(0, 36e6)
        self.np.random.seed(random_state)
        random_list = self.np.random.randint(0, 36e6, size=self.num_samples)
        sample_scores = []
        for sample_ndx in self.tqdm(range(self.num_samples), disable=(not verbose)):
            X_sample, _, y_sample, _ = self.train_test_split(X, y, train_size=self.sample_size, stratify=y,
                                                        random_state=random_list[sample_ndx] + 17)
            X_train, X_test, y_train, y_test = self.train_test_split(X_sample, y_sample, test_size=self.test_size,
                                                                stratify=y_sample, random_state=random_list[sample_ndx])
            clf = model.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            sample_scores.append(self.metric(y_test, y_pred))
            mean = self.np.mean(sample_scores)
        return mean, sample_scores

class RandomSearchResample(RSMixin):

    def __init__(self, model, params, n_iter=60, num_samples=10, sample_size=0.2, test_size=0.3, metric=None,
                 random_state=None):
        self.n_iter = n_iter
        super().__init__(model, params, num_samples, sample_size, test_size, metric, random_state)
        self.param_grid = self._generate_grid()

    def _generate_grid(self):
        return generate_random_grid(self.params, self.n_iter, self.random_state*42)

class GridSearchResample(RSMixin):

    def __init__(self, model, params, num_samples=10, sample_size=0.2, test_size=0.3, metric=None,
                 random_state=None):
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
            random_state = self.np.random.randint(0, 36e6)
        self.np.random.seed(random_state)
        sample_scores = []
        for train_ndx, test_ndx in self.tqdm(self.kfolds(X, y), disable=(not verbose)):
            try:
                x_train, x_test = X.iloc[train_ndx], X.iloc[test_ndx]
                y_train, y_test = y.iloc[train_ndx], y[test_ndx]
            except:
                x_train, x_test = X[train_ndx], X[test_ndx]
                y_train, y_test = y[train_ndx], y[test_ndx]
            clf = model.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            sample_scores.append(self.metric(y_test, y_pred))
        mean = self.np.mean(sample_scores)
        return mean, sample_scores

class RandomSearchCrossval(CVSearchMixin):

    def __init__(self, model, params, n_iter=60, cv=5, stratified=True, shuffle=True, metric=None,
                 random_state=None):
        self.n_iter = n_iter
        super().__init__(model, params, cv, stratified, shuffle, metric, random_state)
        self.param_grid = self._generate_grid()

    def _generate_grid(self):
        return generate_random_grid(self.params, self.n_iter, self.random_state * 42)

class GridSearchCrossval(CVSearchMixin):
    def __init__(self, model, params, cv=5, stratified=True, shuffle=True, metric=None,
                 random_state=None):
        super().__init__(model, params, cv, stratified, shuffle, metric, random_state)
        self.param_grid = self._generate_grid()

    def _generate_grid(self):
        return generate_grid_grid(self.params)

class SimpleSearchMixin(SearchMixin):

    def __init__(self, model, params, val_set, metric=None, random_state=None):
        self.x_val = val_set[0]
        self.y_val = val_set[1]
        super().__init__(model, params, metric, random_state)

    def _eval(self, model, X, y, verbose=False, random_state=None):
        model.fit(X, y)
        score = self.metric(self.y_val, model.predict(self.x_val))
        return score, score

class RandomSearchSimple(SimpleSearchMixin):

    def __init__(self, model, params, val_set, n_iter=60, metric=None, random_state=None):
        self.n_iter = n_iter
        super().__init__(model, params, val_set, metric, random_state)
        self.param_grid = self._generate_grid()

    def _generate_grid(self):
        return generate_random_grid(self.params, self.n_iter, self.random_state * 42)

class GridSearchSimple(SimpleSearchMixin):

    def __init__(self, model, params, val_set, metric=None, random_state=None):
        super().__init__(model, params, val_set, metric, random_state)
        self.param_grid = self._generate_grid()

    def _generate_grid(self):
        return generate_grid_grid(self.params)