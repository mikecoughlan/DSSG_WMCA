import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import precision_recall_curve, f1_score, auc, roc_curve, confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
import modelfunction as mf

DATA_PATH = 'data/processed/'
PLOT_PATH = 'plots/'

CONFIG = {
		'random_int': 123,
		'test_size': 0.3,
		'models': ['NB', 'SVM', 'RF', 'ADA', 'XG'],
		'neural_nets': ['ANN', 'CNN'],
		'input_features': ['postcode', 'constituency', 'msoa_code', 'lsoa_code',
							'local-authority_E06000023', 'local-authority_E06000040', 'local-authority_E07000192',
							'local-authority_E07000194', 'local-authority_E07000196', 'local-authority_E07000218',
							'local-authority_E07000219', 'local-authority_E07000220', 'local-authority_E07000221',
							'local-authority_E07000222', 'local-authority_E07000234', 'local-authority_E08000025',
							'local-authority_E08000026', 'local-authority_E08000027', 'local-authority_E08000028',
							'local-authority_E08000029', 'local-authority_E08000030', 'local-authority_E08000031',
							'local-authority_E09000026', 'floor-height', 'total-floor-area',
							'total_consumption', 'mean_counsumption', 'median_consumption',
							'prop_households_fuel_poor', 'LATITUDE', 'LONGITUDE']
							}

# setting random seed for reporducibility. SK learn uses numpy random seed.
np.random.seed(CONFIG['random_int'])

def loading_and_splitting(target):

	df = pd.read_csv(DATA_PATH+'numerical_individual_columns_data.csv')
	Target = df[target]
	df = df[CONFIG['input_features']]

	if target == 'current-energy-rating':
		X_train, X_test, y_train, y_test, train_latlon, test_latlon = mf.splitting_train_and_test(df, Target, CONFIG['test_size'], change_non_numeric=False)
		predicting = 'epc'
	else:
		X_train, X_test, y_train, y_test, train_latlon, test_latlon = mf.splitting_train_and_test(df, Target, CONFIG['test_size'], change_non_numeric=False)
		predicting = 'mainheat'

	return X_train, X_test, y_train, y_test, predicting, train_latlon, test_latlon

def get_models():
	models = dict()
	n_trees = [10, 50, 100, 500, 1000]
	crit = ['gini', 'entropy', 'log_loss']
	for n in n_trees:
		models[str(n)] = RandomForestClassifier(n_estimators=n, criterion='gini')
	return models
 
# evaluate a give model using cross-validation
def evaluate_model(model, X_train, y_train):
	scores = cross_validate(model, X_train, y_train, cv=5, scoring={'f1':make_scorer(f1_score, average='macro'),
																		'accuracy': make_scorer(accuracy_score)})
	return scores

def RandomForest_tuning(X_train, X_test, y_train, y_test, predicting):
	# get the models to evaluate
	models = get_models()
	# evaluate the models and store results
	results_f1, results_acc, names = list(), list(), list()
	for name, model in tqdm(models.items()):
		scores = evaluate_model(model, X_train, y_train)
		results_f1.append(scores['test_f1'])
		results_acc.append(scores['test_accuracy'])
		names.append(name)
		print('{0}, {1}, {2}, {3}, {4}'.format(name, np.mean(scores['test_f1']), np.std(scores['test_f1']), np.mean(scores['test_accuracy']), np.std(scores['test_accuracy'])))
	# plot model performance for comparison
	fig = plt.figure()
	ax = fig.add_subplot(211)
	plt.boxplot(results_f1, labels=names, showmeans=True)
	plt.title('F1 Macro Scores')
	ax = fig.add_subplot(212)
	plt.boxplot(results_acc, labels=names, showmeans=True)
	plt.title('Total Accuracy Scores')
	plt.savefig('plots/model_tuning/{0}/RandomForest_ntrees_tuning.png'.format(predicting))


def individual_target_columns(y_test, target):

	new_target = pd.get_dummies(y_test)
	print(new_target)


def RandomForest_model(X_train, X_test, y_train, y_test, predicting, version, rating):
	# search for an optimal value of max_depth for Random Forest


	RF = RandomForestRegressor(n_estimators=1000, criterion='squared_error')
	RF.fit(X_train, y_train)


	# with open('models/{0}/RandomForestClassifier_version_{1}_rating_{2}.h5'.format(predicting, str(version), rating), 'wb') as f:
	# 	pickle.dump(RF, f)

	y_pred = RF.predict(X_test)

	return y_pred



def main(target, stage):

	version = 1
	X_train, X_test, y_train, y_test, predicting, train_latlon, test_latlon = loading_and_splitting(target)

	new_y_train = pd.get_dummies(y_train)
	new_y_test = pd.get_dummies(y_test)
	ratings = ['D', 'E']


	# for rating in ratings:
	# 	print('Training the Random Forest for the {0} Rating....'.format(rating))
	# 	y_train = new_y_train[rating]
	# 	y_test = new_y_test[rating]

		# print('Initiating SVM SMOTE....')
		# over = SVMSMOTE()
		# under = RandomUnderSampler()
		# steps = [('o', over), ('u', under)]
		# pipeline = Pipeline(steps=steps)
		# # transform the dataset
		# print('Fitting SMOTE....')
		# X_train_new, y_train = pipeline.fit_resample(X_train, y_train)

	if stage == 'tuning':
		RandomForest_tuning(X_train, X_test, y_train, y_test, predicting)
	if stage == 'eval':
		# input('Is the version number correct ({0})?'.format(version))
		print('Training model....')
		y_pred = RandomForest_model(X_train, X_test, y_train, y_test, predicting, version, rating=None)
		y_pred_save = pd.DataFrame(y_pred)
		y_pred_save.to_csv('outputs/{0}/RandomForestRegression.csv'.format(predicting), index=False)
		# y_pred = pd.read_csv('outputs/{0}/RandomForest_version_{1}.csv'.format(predicting, str(version)))
		# y_pred = y_pred.to_numpy()
		if predicting == 'epc':
			metrics = mf.calculating_heat_metrics(y_test, [y_pred], ['RF_version_{0}'.format(version)])
			print(metrics)

		if predicting == 'mainheat':
			metrics = mf.calculating_heat_metrics(y_test, [y_pred], ['RF_version_{0}'.format(version)])
			print(metrics)


if __name__ == '__main__':

	target = input('heat or epc? ')
	stage = input('tuning or eval? ')

	if target == 'epc':
		target = 'current-energy-efficiency'
	elif target == 'heat':
		target = 'mainheat-description'
	else:
		raise


	main(target, stage)		# calling the main function

	print('It ran. Good job!')


