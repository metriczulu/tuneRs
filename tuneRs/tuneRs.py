class SearchMixin:

    def __init__(self, model, params, num_samples=10, sample_size=0.2, test_size=0.3, metric=None,
                 val_data=None, random_state=None):
        '''

        :param model: model to tune.  Must have .fit() and .predict() methods in the scikit-learn style
        :param params:
        :param num_samples:
        :param sample_size:
        :param test_size:
        :param metric:
        :param val_data:
        :param random_state:
        '''
        import numpy as np
        import copy
        from tqdm.auto import tqdm
        self.tqdm = tqdm
        from sklearn.model_selection import train_test_split
        self.train_test_split = train_test_split
        self.copy = copy
        self.np = np
        self.set_params(model, params, num_samples, sample_size, test_size, metric, val_data, random_state)

    def set_params(self, model, params, num_samples=10, sample_size=0.2, test_size=0.3, metric=None,
                 val_data=None, random_state=None):
        if metric is None:
            from sklearn.metrics import accuracy_score
            self.metric = accuracy_score
        else:
            self.metric = metric
        if val_data is None:
            self.simple_split = False
        else:
            self.X_val = val_data[0]
            self.y_val = val_data[1]
            self.simple_split = True
        self.num_samples = num_samples
        self.sample_size = sample_size
        self.test_size = test_size
        self.best_params_ = None
        self.best_distribution_ = []
        self.model = model
        if random_state is None:
            random_state = self.np.random.randint(0, 36e6)
        self.random_state = random_state
        self.best_score_ = 0.0
        self.best_estimator_ = None
        self.params = params
        self.param_grid = self.__generate_grid()

    def __generate_grid(self, random_state=None):
        return dict()

    def __eval(self, model, X, y, random_state=None, verbose=False):
        if random_state is None:
            random_state = self.np.random.randint(0, 36e6)
        self.np.random.seed(random_state)
        random_list = self.np.random.randint(0, 36e6, size=self.num_samples)
        sample_scores = []
        if self.simple_split:
            model.fit(X, y)
            return self.metric(self.y_val, self.model.predict(self.X_val)), None
        else:
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

    def fit(self, X, y, train_best_estimator=False, verbose=False, super_verbose=False):
        self.np.random.seed(self.random_state)
        random_list = self.np.random.randint(0, 36e6, size=len(self.param_grid))
        for index, random_param in self.tqdm(enumerate(self.param_grid), disable=(not verbose)):
            self.model.set_params(**random_param)
            score, distribution = self.__eval(self.model, X, y, random_state=random_list[index],
                                                   verbose=super_verbose)
            if score > self.best_score_:
                self.best_params_ = random_param
                self.best_score_ = score
                self.best_distribution_ = distribution
        self.best_estimator_ = self.model.set_params(**self.best_params_)
        if train_best_estimator and not self.simple_split:
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
        sns.distplot(self.best_distribution_, color=color, kde_kws = {'color': linecolor, 'linewidth': 3})

class RandomSearchResample(SearchMixin):

    def __init__(self, model, params, num_iter=60, num_samples=10, sample_size=0.2, test_size=0.3, metric=None,
                 val_data=None, random_state=None):
            super().__init__(model, params, num_samples, sample_size, test_size, metric, val_data, random_state)
            self.num_iter = num_iter
            self.param_grid = self.__generate_grid(self.random_state+3)

    def __generate_grid(self, random_state=None):
        if random_state is None:
            random_state = self.np.random.randint(0, 36000)
        self.np.random.seed(random_state)
        random_list = self.np.random.randint(0, 36000, size=self.num_iter)
        param_list = []
        for index in range(self.num_iter):
            param_list.append({key: self.params[key].rvs(1, random_state=random_list[index])[0] for key in self.params})
        return param_list

class GridSearchResample(SearchMixin):

    def __init__(self, model, params, num_samples=10, sample_size=0.2, test_size=0.3, val_data=None, metric=None,
                 random_state=None):
        super().__init__(model, params, num_samples, sample_size, test_size, metric, val_data, random_state)

        self.param_grid = self.__generate_grid()

    def __generate_grid(self):
        from itertools import product
        key_list = []
        param_list = []
        for key in self.params:
            key_list.append(key)
            param_list.append(self.params[key])
        cartesian_product = product(*param_list)
        param_dict_list = []
        param_size = len(key_list)
        for params in cartesian_product:
            temp_dict = {key_list[i]: params[i] for i in range(param_size)}
            param_dict_list.append(temp_dict)
        return param_dict_list