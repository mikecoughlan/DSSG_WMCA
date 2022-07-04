##################################################################################################
#
# DSSG_WMCA/scripts/data_preprocessing/data_preprocessing.py
#
# pulls together all the preprocessing steps and returns a final dataframe
#
#
##################################################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas_profiling import ProfileReport
import json
import pickle
import requests
from scipy.stats import pearsonr
from tqdm import tqdm
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from CHAID import Tree
from shapely.geometry import Point  #Polygon
import geopandas

import inspect
import gc


DATA_PATH = 'data/processed/'
PLOT_PATH = 'plots/'

CONFIG = {'n_estimators':10,
		'random_int': 123,
		'test_size': 0.3,
		}

# setting random seed for reporducibility. SK learn uses numpy random seed.
np.random.seed(CONFIG['random_int'])

def splitting_train_and_test(df, target, test_size):
	'''Splits the sequentially saved csv file into the training and testing set. 
		File had testing set concat on the training set, so sequentioal split is performed.

	INPUTS:
		df (pd.df): processed data frame with the bottom (test_size %) being the test set.
		target (str): columns name for the target array.
		test_size (float between 0 and 1): portion of the dataset that will be segmented for testing.

	RETURNS:
		X_train (np.array): input training array for fitting the model.
		X_test (np.array): input testing array to predict the fit model on.
		y_train (np.array): target array for fitting the model.
		y_test (np.array): ground truth for comparing the models predictions.'''

	y = target
	print(target)
	y_train = y[:int((len(df)*(1-test_size)))]
	print(y_train)
	y_test = y[int((len(df)*(1-test_size))):]
	print(y_test)
	X_train = df[:int((len(df)*(1-test_size)))]
	X_test = df[int((len(df)*(1-test_size))):]

	print(y_train.isnull().sum())
	print(y_test.isnull().sum())
	print(X_train.isnull().sum())
	print(X_test.isnull().sum())

	return X_train, X_test, y_train, y_test



def RandomForestModel(X_train, X_test, y_train, y_test, n_estimators=None, random_int=None):
	'''Initalizes and fits a Random Forest model and makes a prediction on the testing data. 

		INPUTS:
			X_train (np.array): input training array for fitting the model.
			X_test (np.array): input testing array to predict the fit model on.
			y_train (np.array): target array for fitting the model.
			y_test (np.array): ground truth for comparing the models predictions.
			n_estimators (int): The number of trees in the forest.
			random_int (int): integer that defines the random state for reproduction

		RETURNS:
			model: fitted random forest model.
			'''

	model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_int)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	acc = accuracy_score(y_test, y_pred)
	print('Random Forest Accuracy Score: '+str(acc))

	return model


def plotting_most_important_features(model, feature_names):

	feature_importance = model.feature_importances_

	fig = plt.figure(figsize=(20,10))

	plt.barh(feature_names, feature_importance)
	plt.xlabel("Random Forest Feature Importance")

	plt.savefig(PLOT_PATH+'random_forest_feature_importance.png')



def main(config):

	df = pd.read_csv(DATA_PATH+'numerical_data.csv')
	print(df.columns)
	df.drop(['Unnamed: 0', 'Unnamed: 0.1','Unnamed: 0.1.1'], inplace=True, axis=1)
	target = df['current-energy-rating']
	df = df.select_dtypes(exclude=['object'])
	print(df.columns)

	X_train, X_test, y_train, y_test = splitting_train_and_test(df, target, config['test_size'])

	model = RandomForestModel(X_train, X_test, y_train, y_test, n_estimators=10, random_int=config['random_int'])

	plotting_most_important_features(model, X_train.columns.tolist())


if __name__ == '__main__':

	main(CONFIG)

	print('It ran. Good job!')










