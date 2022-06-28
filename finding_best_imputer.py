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


def extracting_numeric(df):
	# seperating the EPC column
	EPC = df['current-energy-rating']

	# removing non-numeric data 
	df = df.select_dtypes(exclude=['object'])

	# adding back in the EPC data for splitting into catagories
	df['current-energy-rating'] = EPC

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



def splitting_df_by_EPC(df):
	'''Splits the larger dataframe into a data frame for each 
		EPC rating. Done for the simple_inputing

		INPUTS:
			EPC_list (pd.df or list of pd.df): list of dataframes. One for each EPC rating.

		RETURNS:
			EPC_list (list of dfs): list of dfs for each EPC rating.
	'''
	EPC_list = [] # establishing the list

	# looping through each EPC rating
	for EPC in df['current-energy-rating'].unique():
		EPC_list.append(df[df['current-energy-rating']==EPC]) # appends the new dataframe to a list

	return EPC_list



def simple_imputing(train, test, features):
	'''Does the simple imputing of the data. Takes
		in the data seperated by EPC rating and calculates
		teh skew for the columns with missing data.
		If the skew is high (above 0.5), the median
		value is used for imputing. If the skew is
		low, the mean value is used. 

		INPUTS:
			train (pd.df or list of pd.df): train dataframe.
			features (str or list of str): features that need values filled.

		RETURNS:
			train (pd.df): combined dataframe with all EPC ratings for the train data including imputed values.
			test (pd.df): combined dataframe with all EPC ratings for the testing data including imputed values calculated from training dataset.
	'''

	Train = splitting_df_by_EPC(train)		# splitting the dataframe into one for each EPC rating
	Test = splitting_df_by_EPC(test)		# splitting the dataframe into one for each EPC rating

	print(Train[0].columns)

	# looping through all EPC ratings
	for train_EPC, test_EPC in zip(Train, Test):

		# looping through all the features that are missing data
		for feature in features:

			train_data = train_EPC[feature]
			test_data = test_EPC[feature]
			skew = train_data.skew(axis=0, skipna = True)  # finds the skew of the data
			train_data = train_data.to_numpy().reshape(-1,1)
			test_data = test_data.to_numpy().reshape(-1,1)

			# if the skew is low, we impute values with the mean of the non-missing data
			if abs(skew) <= 0.5:
				imp = SimpleImputer(missing_values=np.nan, strategy='mean')
				imp.fit(train_data)
				train_EPC[feature] = imp.transform(train_data)
				test_EPC[feature] = imp.transform(test_data)

			# if the skew is high, we impute with the median of the non-missing data
			else:
				imp = SimpleImputer(missing_values=np.nan, strategy='median')
				imp.fit(train_data)
				train_EPC[feature] = imp.transform(train_data)
				test_EPC[feature] = imp.transform(test_data)


	train = pd.concat(Train, axis=0)	# concatinating all of the dataframes together
	test = pd.concat(Test, axis=0)	# concatinating all of the dataframes together

	train = train.sample(frac=1, random_state=random_int).reset_index(drop=True)
	test = test.sample(frac=1, random_state=random_int//2).reset_index(drop=True)

	return train, test



def iterative_imputing(df, iterative_imputing_columns, target, random_state=None):
	'''Does the simple imputing of the data. Takes
		in the data seperated by EPC rating and calculates
		teh skew for the columns with missing data.
		If the skew is high (above 0.5), the median
		value is used for imputing. If the skew is
		low, the mean value is used. 

		INPUTS:
			df (pd.df): pandas dataframe containing numerical data needing imputing.
			random_state (int): random state to use for reproducibility.

		RETURNS:
			df (pd.df): combined dataframe with all EPC ratings.
	'''

	# X = numerical[iterative_imputing_columns]		# seperating the EPC ratings from the larger dataset
	X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.3, random_state=random_int, stratify=target)
	cat_train = X_train.drop(iterative_imputing_columns, axis=1)
	cat_test = X_test.drop(iterative_imputing_columns, axis=1)

	catagorical = pd.concat([cat_train, cat_test], axis=0).reset_index(drop=True)
	print(catagorical)

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

	print(numeric)

	df = pd.concat([catagorical, numeric], axis=1, ignore_index=False)
	print(df)

	return df


def KNN_imputing(train, test, n_neighbors=5, weights='uniform', metric='nan_euclidean'):
	'''Does the simple imputing of the data. Takes
		in the data seperated by EPC rating and calculates
		teh skew for the columns with missing data.
		If the skew is high (above 0.5), the median
		value is used for imputing. If the skew is
		low, the mean value is used. 

		INPUTS:
			df (pd.df): pandas dataframe containing numerical data needing imputing.
			n_neighbors (int): Number of neighboring samples to use for imputation.
			weights: Weight function used in prediction. 

		RETURNS:
			df (pd.df): combined dataframe with all EPC ratings.
	'''

	imp = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric=metric)  # initalizing the imputer
	imp.fit(train) 																# fitting the imputer	
	train = imp.transform(train)														# transforming the df on the fit imputer
	test = imp.transform(test)														# transforming the df on the fit imputer

	return train, test


def random_forest(X_train, X_test, y_train, y_test, random_state=None):

	model = RandomForestClassifier(n_estimators=10)
	# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=random_state)
	# scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, error_score='raise')
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	print(y_pred)
	print(np.unique(y_pred))
	score = accuracy_score(y_test.to_numpy(), y_pred)



	return score


def splitting_X_and_y(df):
	'''splitting the training data from the target.'''

	target = df['current-energy-rating']
	df.drop('current-energy-rating', inplace=True, axis=1)

	return df, target



def main():
	'''main function'''

	print('loading csv...')
	df = load_csv()

	print(df)

	# dropping just the rows that have missing or invalid EPC ratings
	df = df[df['current-energy-rating'].notna()] # dropping columns where there is no EPC rating


	print('extracting numeric...')
	# numeric = df.select_dtypes(exclude=['object'])
	# numeric_columns = numeric.columns


	# removing non-numeric data 
	numeric = df.select_dtypes(exclude=['object'])
	print('dropping unnecessary columns...')
	numeric = dropping_unnecessary_numeric_columns(numeric)
	numeric_columns = numeric.columns

	catagorical = df.select_dtypes(exclude=['float', 'int'])
	catagorical_columns = catagorical.columns

	iterative_imputing_columns = extracting_columns_based_on_missing_percentage(numeric, percent_range=[0,80], percent_equal=None)


	# numeric['current-energy-rating'] = EPC

	# X, y = splitting_X_and_y(numeric.copy())

	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_int, stratify=y)
	# X_train['current-energy-rating'] = y_train
	# X_test['current-energy-rating'] = y_test

	# results_dict = {} # establishing a dict to store results in.


	# print('Initiating Simple Imputing....')
	# simple_train_df, simple_test_df = simple_imputing(X_train, X_test, simple_imputing_columns)


	# print('Splitting....')
	# simple_X_train, simple_y_train = splitting_X_and_y(simple_train_df)
	# simple_X_test, simple_y_test = splitting_X_and_y(simple_test_df)


	# print("Random Classifier....")
	# simple_accuracy_score = random_forest(simple_X_train, simple_X_test, simple_y_train, simple_y_test, random_state=random_int)


	# print('Simple_accuracy_score: '+str(simple_accuracy_score))
	# results_dict['simple'] = simple_accuracy_score


	# simple_X_train['current-energy-rating'] = simple_y_train
	# simple_X_test['current-energy-rating'] = simple_y_test


	# simple_X_train.to_csv('../data/processed/simple_numerical_processed_training.csv')
	# simple_X_test.to_csv('../data/processed/simple_numerical_processed_testing.csv')

	# del simple_X_train, simple_X_test, simple_y_train, simple_y_test
	# gc.collect()



	print('Initiating Iterative Imputing...')
	imputed_df = iterative_imputing(df, iterative_imputing_columns, df['current-energy-rating'], random_state=random_int)
	
	print(imputed_df.columns)

	
	# print('Initiating KNN Imputing...')
	# KNN_X_train, KNN_X_test = KNN_imputing(X_train, X_test)
	# print("Random Classifier")
	# KNN_accuracy_score = random_forest(KNN_X_train, KNN_X_test, y_train, y_test, random_state=random_int)
	

	# print('KNN_accuracy_score: '+str(KNN_accuracy_score))
	# results_dict['simple'] = KNN_accuracy_score
	
	# del KNN_X_train, KNN_X_test
	# gc.collect()

	

	# print("Random Classifier")
	# KNN_mean, KNN_std = random_forest(KNN_X, KNN_y, random_state=random_int)
	# print('KNN mean: '+str(KNN_mean) + '| KNN STD: '+str(KNN_std))
	# results_dict['KNN'] = {'mean:':KNN_mean, 'std':KNN_std}

	# del KNN_X, KNN_y
	# gc.collect()


	# with open('../outputs/imputer_test_dict.pkl') as f:
	# 	pickle.dump(results_dict, f)

	imputed_df.to_csv('../data/processed/fully_imputed_EPC_elec_consump_fuel_pov_data.csv')



if __name__ == '__main__':

	main()

	print('It ran! Good job.')




