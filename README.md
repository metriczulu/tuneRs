# Hyperparameter TuneRs

**tuneRs** is a small package for tuning hyperparameters.  Originally just focusing on resampling methods (hence the 'Rs' for 'Resample'), it has been extended to cover a range of hyperparameter tuning methods.  The current package supports resampling, crossvalidation, and simple train/val split to estimate the effectiveness of each set of hyperparameters.  On top of this, each of those said methods can be applied with a grid search over parameters, a random search over paramaters, or a mix of both.  Estimating model accuracy using resampling methods is much quicker that using k-fold crossvalidation--although resampling tends to underestimate accuracy more that crossvalidation.  Resampling underestimates accuracy in a *consistent* fashion, however, which still makes it valuable for tuning hyperparameters.  Due to it's consistency, choosing hyperparameters based on aggregated samples still gets within the neighborhood of maximal while being much, much faster.  This is a package to help you get there.

**tuners.ResampleSearch** uses resampling to estimate the accuracy of a model.  Currently supports grid search, random search, and a mix of the two.

**tuners.CrossvalSearch** uses K-fold crossvalidation to estimate the accuracy of a model.  Currently supports grid search, random search, and a mix of the two.

**tuners.SimpleSearch** uses a simple train/validation split to estimate the accuracy of a model.  Currently supports grid search, random search, and a mix of the two.  This is, by far, the fastest search method in the package but also the least reliable.  Final hyperparameters are heavily dependent on the specific training and validation sets used.

The **space** module provides various distributions that can be used to select parameters in a random search.  In addition to this, the package supports **skopt.space** objects for random parameter distributions as well.

## Current Version is v0.6

This package is currently in the beginning stages and is very bare-bones

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tuneRs.

```bash
pip install tuneRs
```

## Usage

All classes are meant to mimic the scikit-learn tuners (to a certain degree).  A simple example would be:

```python	
from sklearn.svm import SVC
from tuneRs import tuners, space
	
model = SVC()
	
# Grid search parameters
parameters = {"gamma": [0.001, 0.01, 0.1, 1.0, 10.0],
			"C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
			"degree": [1, 2, 3],
			"kernel": ['poly', 'rbf']}

# Set up for a grid hyperparameter search with resample accuracy estimate
tuner = tuners.ResampleSearch(model, grid_params=parameters, sample_size=0.3, num_samples=12)

# Fit the tuner
tuner.fit(X_train, y_train)
	
# Display the best parameters found
tuner.best_params_
	
# Display the aggregate resample score of the best parameters <br/>
tuner.best_score_
	
# Define our new model
model = tuner.best_estimator_
	
# Plot the resample accuracy distribution for the model with best hyperparameters <br/>
tuner.plot_best()
	
# Random search parameters
parameters = {"gamma": space.LogNormal(0.001, 10.0, dtype="float32"),
			"C": space.LogNormal(0.01, 1000.0, dtype="float32"),
			"degree": space.Uniform(1, 4, dtype="int"),
			"kernel": space.Categorical(['poly', 'rbf'], probs=[0.75, 0.25])}
# Set up a random hyperparameter search over 100 parameter combinations with 5-fold crossvalidation accuracy estimate
tuner = tuners.CrossvalSearch(model, random_params=parameters, n_random=100, cv=5)
	
# Proceed as above for everything else
tuner.fit(X_train, y_train)
print(f"The best parameters found with the random hyperparameter search are {tuner.best_params_}")
```
	
## Future Plans

Multiple tuners are currently planned to be added.  The next one will be a Bayesian search method.  A dynamic version of grid search and random search is currently being worked on that iterates fit multiple times on increasingly small areas of the data space.

## License

Lol
