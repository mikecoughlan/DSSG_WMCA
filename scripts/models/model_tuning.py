##############################################################################
#
#
#	DSSG_WMCA/scripts/models/model_tuning.py
#
#	Does iteritive model tuning using cross validation. Looks at F1 Macro score
# 	and the overall accuracy score and plots boxplots with the results.
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
from naive_bayes import NaiveBayes_tuning
from SVM import SVC_tuning
from RF import RandomForest_tuning
from ADABoost import ADABoost_tuning
from XGBoost import XGBoost_tuning
from ANN import ANN_tuning
from CNN import CNN_tuning

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
							'local-authority_E09000026', 'floor-height', 'total-floor-area', 'energy-consumption-current',
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
		predicting = 'epc'
	else:
		X_train, X_test, y_train, y_test = mf.splitting_train_and_test(df, Target, CONFIG['test_size'], change_non_numeric=False)
		predicting = 'mainheat'

	scaled_X_train, scaled_X_test, scaler = mf.creating_scaler_for_NN(X_train, X_test)
	
	# getting the models and the predicted columns. Going to have to import these once the files are written.
	
	NaiveBayes_tuning(X_train, X_test, y_train, y_test, predicting)

	SVC_tuning(X_train, X_test, y_train, y_test, predicting)

	Random_Forest_tuning(X_train, X_test, y_train, y_test, predicting)

	ADABoost_tuning(X_train, X_test, y_train, y_test, predicting)

	XGBoost_model(X_train, X_test, y_train, y_test, predicting)




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





















