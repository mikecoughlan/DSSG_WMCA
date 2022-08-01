##############################################################################
#
#
#	DSSG_WMCA/scripts/models/modeling_pipeline.py
#
#	The full modeling pipeline. Will contain config files and one main function
#	that pulls in all other functions to test all the models against each other.
# 	Will result in diffferent saved models and evaluation metrics for comparing
# 	the models against each other.
#
#
##############################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas_profiling import ProfileReport
import json
import pickle
import requests
import argparse
from scipy.stats import pearsonr
from tqdm import tqdm
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import precision_recall_curve, f1_score, auc, roc_curve, confusion_matrix

from tensorflow.keras.utils import to_categorical
import modelfunction as mf
# from naive_bayes import *
# from SVM import *
from RF import *
from ADABoost import *
from XGBoost import *
# from ANN import *
# from CNN import *
from plotting_metrics import *

import inspect
import gc


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
							'local-authority_E09000026',
							'floor-height', 'total-floor-area',
							'total_consumption', 'mean_counsumption', 'median_consumption',
							'prop_households_fuel_poor', 'LATITUDE', 'LONGITUDE']
							}

# setting random seed for reporducibility. SK learn uses numpy random seed.
np.random.seed(CONFIG['random_int'])


def main(target):

	df = pd.read_csv(DATA_PATH+'numerical_individual_columns_data.csv')
	Target = df[target]
	df = df[CONFIG['input_features']]

	if target == 'current-energy-rating':
		X_train, X_test, y_train, y_test = mf.splitting_train_and_test(df, Target, CONFIG['test_size'], change_non_numeric=True)
	else:
		X_train, X_test, y_train, y_test = mf.splitting_train_and_test(df, Target, CONFIG['test_size'], change_non_numeric=False)
	
	# getting the models and the predicted columns. Going to have to import these once the files are written.
	
	# NB_pred = NaiveBayes_model()

	# SVM_pred = SVM_model()

	# RF_pred = Random_Forest_model(X_train, X_test, y_train, y_test)

	# print(RF_pred)
	# RF = pd.DataFrame(RF_pred)
	# RF.to_csv('outputs/RF_epc_pred.csv', index=False)

	# ADA_pred = ADABoost_model(X_train, X_test, y_train, y_test)

	# print(ADA_pred)
	# ADA = pd.DataFrame(ADA_pred)
	# ADA.to_csv('outputs/ADA_epc_pred.csv', index=False)

	# XGB_pred = XGBoost_model(X_train, X_test, y_train, y_test)

	# print(XGB_pred)
	# XGB = pd.DataFrame(XGB_pred)
	# XGB.to_csv('outputs/XGB_epc_pred.csv', index=False)

	if target == 'mainheat-description':
		folder = 'mainheat'
	if target == 'current-energy-rating':
		folder = 'epc'

	# naive = pd.read_csv('outputs/{0}/knn_naive_model.csv'.format(folder))
	# NB_pred = pd.read_csv('outputs/{0}/NaiveBayes_version_0.csv'.format(folder))
	# SVC_pred = pd.read_csv('outputs/{0}/SVC_version_0.csv'.format(folder))
	# RF_pred = pd.read_csv('outputs/{0}/RandomForest_version_1.csv'.format(folder))
	# ADA_pred = pd.read_csv('outputs/{0}/ADABoost_version_0.csv'.format(folder))
	# XGB_pred = pd.read_csv('outputs/{0}/XGBoost_version_5.csv'.format(folder))

	A = pd.read_csv('outputs/epc/RandomForest_version_1_rating_A.csv'.format(folder))
	B = pd.read_csv('outputs/epc/RandomForest_version_1_rating_B.csv'.format(folder))
	C = pd.read_csv('outputs/epc/RandomForest_version_1_rating_C.csv'.format(folder))
	D = pd.read_csv('outputs/epc/RandomForest_version_1_rating_D.csv'.format(folder))
	E = pd.read_csv('outputs/epc/RandomForest_version_1_rating_E.csv'.format(folder))
	F = pd.read_csv('outputs/epc/RandomForest_version_1_rating_F.csv'.format(folder))
	G = pd.read_csv('outputs/epc/RandomForest_version_1_rating_G.csv'.format(folder))

	# naive = naive.to_numpy()
	# NB_pred = NB_pred.to_numpy()
	# SVC_pred = SVC_pred.to_numpy()
	# RF_pred = RF_pred.to_numpy()
	# ADA_pred = ADA_pred.to_numpy()
	# XGB_pred = XGB_pred.to_numpy()

	A = A.to_numpy()
	B = B.to_numpy()
	C = C.to_numpy()
	D = D.to_numpy()
	E = E.to_numpy()
	F = F.to_numpy()
	G = G.to_numpy()

	new_y_test = pd.get_dummies(y_test)

	pred = [A, B, C, D, E, F, G]
	names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

	test = []
	for name in names:
		test.append(new_y_test[name])

	if target == 'current-energy-rating':
		metrics = mf.calculating_epc_metrics(test, pred, names)

		plotting_epc_metrics(metrics)
	else:
		metrics = mf.calculating_heat_metrics(test, pred, names)
		plotting_mainheat_metrics(metrics)

	metrics.to_csv('outputs/{0}/initial_random_forest_metrics.csv'.format(folder))



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('target',
						action='store',
						choices=['epc', 'heat'],
						type=str,
						help='choose which target param to examine. Type epc for current-energy-rating, and heat for mainheat-description')

	
	args=parser.parse_args()

	if args.target == 'epc':
		target = 'current-energy-rating'
	elif args.target == 'heat':
		target = 'mainheat-description'
	else:
		raise

	main(target)		# calling the main function

	print('It ran. Good job!')





















