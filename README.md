# Hyperparameter TuneRs

**tuneRs** is a small package for tuning hyperparameters using resampling methods instead of normal crossvalidation.  Estimating model accuracy using resampling methods is much quicker that using k-fold crossvalidation--although resampling tends to underestimate accuracy more that crossvalidation.  Resampling underestimates accuracy in a *consistent* fashion, however, which still makes it valuable for tuning hyperparameters.  Due to it's consistency, choosing hyperparameters based on aggregated samples still gets within the neighborhood of maximal while being much, much faster.  This is a package to help you get there.

**GridSearchResample** uses the grid search method to optimize hyperparameters.

**RandomSearchResample** uses the random search method to optimize hyperparameters.
## Current Version is v0.54

This package is currently in the beginning stages and is very bare-bones

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tuneRs.

```bash
pip install tuneRs
```

## Usage

Both classes are meant to mimic the scikit-learn tuners (to a certain degree).  A simple example would be:

	import tuneRs
	
	model = SVC(kernel='rbf')
	
	parameters = {["gamma": [0.001, 0.01, 0.1, 1.0, 10.0],
			"C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}

	# Set up for a random hyperparameter search
	tuner = tuneRs.RandomSearchResample(model, parameters, num_iters=300, sample_size=0.3, num_samples=12)

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

## Future Plans

Multiple tuners are currently planned to be added.  The next one will be a Bayesian search method.  A dynamic version of grid search and random search is currently being worked on that iterates fit multiple times on increasingly small areas of the data space.

## License

Lol
