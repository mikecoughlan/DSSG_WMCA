##################################################################################################
#
# DSSG_WMCA/scripts/models/randomforest.py
#
# runs a random forest model on the processed data
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import precision_recall_curve, f1_score, auc, roc_curve

from tensorflow.keras.utils import to_categorical

import modelfunction as mf

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

	model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_int)		# initilizes the RF model
	model.fit(X_train, y_train)																# fits the RF model


	y_pred = model.predict_proba(X_test)													# does a probabilistic prediction
	
	___ = mf.calculating_epc_metrics(y_test, [y_pred], 'current-energy-rating')													# Calculates the metrics using a function from a different file

	acc = accuracy_score(y_test, y_pred)													# getting the accuracy score
	print('Random Forest Top {0} Columns Accuracy Score: '.format(len(X_train.columns))+str(acc))

	return model, acc 						# returns the model and the accuracy score



def plotting_most_important_features(model, feature_names):

	feature_importance = model.feature_importances_						# getting the most important features from the model

	FI_df = pd.DataFrame({'features':feature_names,
							'importance':feature_importance})			# putting the most important features into a data frame 

	FI_df.sort_values(by=['importance'], ascending=False, axis=0, inplace=True)		# putting the features in order

	FI_df = FI_df[:60]							# taking the top 60 to plot

	fig = plt.figure(figsize=(20,10))			# plotting the most important features

	plt.barh(FI_df['features'], FI_df['importance'])
	plt.xlabel("Random Forest Feature Importance")

	plt.savefig(PLOT_PATH+'random_forest_feature_importance_mainheat_target.png')



def main(config):

	df = pd.read_csv(DATA_PATH+'numerical_individual_columns_data.csv')
	df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)
	df[df.select_dtypes(np.int64).columns] = df.select_dtypes(np.int64).astype(np.int32)


	target = df['current-energy-rating']			# defining the target variable
	df.drop(['current-energy-rating', 'main-heating-controls', 'mainheat-description'], axis=1, inplace=True)		# dropping the tagert from the input
	df = df.select_dtypes(exclude=['object'])

	trimmed_cols = ['percentage-low-energy-lighting', 'LONGITUDE', 'LATITUDE',
					'prop_households_fuel_poor', 'median_consumption', 'mean_counsumption', 'total_consumption', 
					'extension-count', 'lighting-cost-current', 'lodgement-datetime',
					'multi-glaze-proportion', 'photo-supply', 'windows-energy-eff', 'number-habitable-rooms', 
					'total-floor-area', 'inspection-date', 'glazed-area', 'number-open-fireplaces', 'floor-height',
					'photo-supply-binary_True', 'photo-supply-binary_False', 'walls-description_node9', 'walls-description_node8',
					'walls-description_node7', 'walls-description_node6', 'walls-description_node5', 'walls-description_node4',
					'walls-description_node3', 'walls-description_node2', 'walls-description_node1', 'tenure_rental (social)',
					'tenure_rental (private)', 'tenure_owner-occupied', 'transaction-type_rental (social)', 'transaction-type_rental (private)',
					'transaction-type_rental', 'transaction-type_not sale or rental', 'transaction-type_none of the above', 
					'transaction-type_non marketed sale', 'transaction-type_new dwelling', 'transaction-type_marketed sale',
					'transaction-type_following green deal', 'transaction-type_assessment for green deal', 'transaction-type_Stock condition survey',
					'transaction-type_RHI application', 'transaction-type_FiT application', 'transaction-type_ECO assessment',
					'roof-description_node10', 'roof-description_node9', 'roof-description_node8', 'roof-description_node7',
					'roof-description_node6', 'roof-description_node5', 'roof-description_node4', 'roof-description_node3',
					'roof-description_node2', 'roof-description_node1', 'windows-description_node4', 'windows-description_node3',
					'windows-description_node2', 'windows-description_node1', 'built-form_Semi-Detached', 'built-form_Mid-Terrace',
					'built-form_End-Terrace', 'built-form_Enclosed Mid-Terrace', 'built-form_Enclosed End-Terrace', 'built-form_Detached',
					'local-authority_E09000026', 'local-authority_E08000031', 'local-authority_E08000030', 'local-authority_E08000029',
					'local-authority_E08000028', 'local-authority_E08000027', 'local-authority_E08000026', 'local-authority_E08000025',
					'local-authority_E07000234', 'local-authority_E07000222', 'local-authority_E07000221', 'local-authority_E07000220',
					'local-authority_E07000219', 'local-authority_E07000218', 'local-authority_E07000196', 'local-authority_E07000194',
					'local-authority_E07000192', 'local-authority_E06000040', 'local-authority_E06000023', 'floor-description_node7',
					'floor-description_node6', 'floor-description_node5', 'floor-description_node4', 'floor-description_node3', 
					'floor-description_node2', 'floor-description_node1', 'mechanical-ventilation_natural', 'mechanical-ventilation_mechanical, supply and extract',
					'mechanical-ventilation_mechanical, extract only', 'property-type_Park home', 'property-type_Maisonette', 'property-type_House',
					'property-type_Flat', 'property-type_Bungalow', 'glazed-type_triple, known data', 'glazed-type_triple glazing', 
					'glazed-type_single glazing', 'glazed-type_secondary glazing', 'glazed-type_double, known data', 'glazed-type_double glazing, unknown install date',
					'glazed-type_double glazing installed during or after 2002', 'glazed-type_double glazing installed before 2002', 'lighting-description',
					'lsoa_code', 'msoa_code', 'constituency', 'postcode']


	df = df[trimmed_cols]										# trimming columns

	df.fillna(df.mean(), inplace=True)

	X_train, X_test, y_train, y_test = mf.splitting_train_and_test(df, target, config['test_size'])		# external function to split the data into test/train

	model, acc = RandomForestModel(X_train, X_test, y_train, y_test, n_estimators=10, random_int=config['random_int'])		# calling the RF model and returning the score and trained model
	
	accuracy.append(acc)

	# plotting_most_important_features(model, X_train.columns.tolist())		# calling the plotting function



if __name__ == '__main__':

	main(CONFIG)		# calling the main function

	print('It ran. Good job!')










