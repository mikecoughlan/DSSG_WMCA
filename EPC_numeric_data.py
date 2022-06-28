##################################################################################################
#
# DSSG_WMCA/scripts/EPC_numeric_data.py
#
# Script to examine the numeric EPC data, remove columns with large amounts of
# missing data, and impute others. Will test several imputation methods and 
# test them using a random forrest classifier.
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
from sklearn.metrics import accuracy_score
import gc


# setting random seed for reporducibility. SK learn uses numpy random seed.
random_int = 123
np.random.seed(random_int)


def load_csv():
	# loading preprocessed data. This is currently hardcoded. Clean this up.
	df = pd.read_csv('../data/processed/chaid_data.csv')
	if 'Unnamed: 0' in df.columns:
		df.drop('Unnamed: 0', inplace=True, axis=1)

	return df



def dropping_unnecessary_numeric_columns(df):
	# Dropping columns with more than 90% missing data
	drop_missing = []
	percent_missing = df.isnull().sum() * 100 / len(df)
	for feat, missing in zip(percent_missing.index, percent_missing):
		if missing >= 98:
			drop_missing.append(feat)
	df.drop(drop_missing, inplace=True, axis=1)


	# dropping a few more parameters that we don't expect to be relevent
	# including current and potential energy efficiency to drop as well
	# also currently hardcoded. Fix.
	to_drop = ['num_households_fuel_poverty', 'num_households', 'current-energy-efficiency',
				'environment-impact-potential', 'environment-impact-current', 'co2-emissions-current', 'uprn']
	df.drop(to_drop, inplace=True, axis=1)

	return df


def extracting_columns_based_on_missing_percentage(df, percent_range=None, percent_equal=None):
	'''extracts column names from a dataframe based on the percentage of data they are missing from the 
		larger dataframe.

		INPUTS:
			df: larger pandas dataframe containing all the data
			 percent_range [lower, upper]: lower and upper missing percentage values (inclusive) to be included 
			 								in the extracted column list. 
			 percent_equal (int or float): number for an exact percentage to be used for extracting columns.

		RETURNS:
			columns: list of column names that fall withing the percent missing bounds specified.
		'''

	columns = []
	percent_missing = df.isnull().sum() * 100 / len(df)
	if percent_range:	
		for feat, missing in zip(percent_missing.index, percent_missing):
			if (missing >= percent_range[0]) and (missing <= percent_range[1]):
				columns.append(feat)

	if percent_equal:
		for feat, missing in zip(percent_missing.index, percent_missing):			
			if missing == percent_range[0]:
				columns.append(feat)


	return columns




def iterative_imputing(df, iterative_imputing_columns, target, random_state=None):
	'''Does the simple imputing of the data. Takes
		in the data seperated by EPC rating and calculates
		teh skew for the columns with missing data.
		If the skew is high (above 0.5), the median
		value is used for imputing. If the skew is
		low, the mean value is used. 

		INPUTS:
			df (pd.df): pandas dataframe containing all data needing imputing.
			iterative_imputing_columns (str, or list of str): list of columns for which the imputing will be performed. Must be numeri columns.
			target (pd.Series): target variable for the model.
			random_state (int): random state to use for reproducibility.

		RETURNS:
			df (pd.df): combined dataframe containing the imputed data.
	'''

	# X = numerical[iterative_imputing_columns]		# seperating the EPC ratings from the larger dataset
	X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.3, random_state=random_int, stratify=target)
	cat_train = X_train.drop(iterative_imputing_columns, axis=1)
	cat_test = X_test.drop(iterative_imputing_columns, axis=1)

	catagorical = pd.concat([cat_train, cat_test], axis=0).reset_index(drop=True)

	X_train = X_train[iterative_imputing_columns]
	X_test = X_test[iterative_imputing_columns]
	
	imp = IterativeImputer(verbose=1, random_state=random_state)	# initalizing the imputer
	imp.fit(X_train)														# fitting the imputer on the dataframe
	X_train = imp.transform(X_train)											# transforming the df on the fit imputer
	X_test = imp.transform(X_test)											# transforming the df on the fit imputer

	print("Random Classifier...")
	iterative_accuracy_score = random_forest(X_train, X_test, y_train, y_test, random_state=random_int)
	print('Iterative Accuracy Score: '+str(iterative_accuracy_score))

	X = np.concatenate((X_train, X_test),axis=0)

	print('Transforming numerical data...')
	numeric = imp.transform(X)

	numeric = pd.DataFrame(numeric, columns=iterative_imputing_columns)


	df = pd.concat([catagorical, numeric], axis=1, ignore_index=False)


	return df



def random_forest(X_train, X_test, y_train, y_test, random_state=None):

	model = RandomForestClassifier(n_estimators=10)
	# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=random_state)
	# scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, error_score='raise')
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	score = accuracy_score(y_test.to_numpy(), y_pred)

	return score



def main():
	'''main function'''

	print('loading csv...')
	df = load_csv()

	# dropping just the rows that have missing or invalid EPC ratings
	df = df[df['current-energy-rating'].notna()] # dropping columns where there is no EPC rating

	# removing non-numeric data 
	numeric = df.select_dtypes(exclude=['object'])
	print('dropping unnecessary columns...')
	numeric = dropping_unnecessary_numeric_columns(numeric)
	numeric_columns = numeric.columns

	iterative_imputing_columns = extracting_columns_based_on_missing_percentage(numeric, percent_range=[0,80], percent_equal=None)

	print('Initiating Iterative Imputing...')
	imputed_df = iterative_imputing(df, iterative_imputing_columns, df['current-energy-rating'], random_state=random_int)


	imputed_df.to_csv('../data/processed/fully_imputed_EPC_elec_consump_fuel_pov_data.csv')



if __name__ == '__main__':

	main()

	print('It ran! Good job.')




