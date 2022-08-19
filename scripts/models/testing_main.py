##############################################################################
#
#
#	DSSG_WMCA/scripts/models/testing_main.py
#
#	The full modeling pipeline. Starts by taking the fully processed EPC data
# 	and merging it with the proxy data to get the 'calculatedAreaValue' and
# 	'ResHMax' values, which are proxies for the total-floor-area and the 
# 	floor height. Then passes the data through the similarity quantification 
# 	model which will identify homes with the same floor area footprint as
# 	homes in the EPC database within the same target area (postcode, lsoa, etc.)
# 	and assigns the EPC rating to the similar home. The homes with similar
# 	homes with EPC ratings and those without are seperated. A Random Forest (RF) 
# 	model is run on both datasets. For the non-similar homes the RF results are 
# 	used as the EPC results for that home, for the similar homes set, the 
# 	RF and SQ results are compared and if they do not match, the confidence
# 	level of the RF is examined. If it exceeded 0.5 the RF value is used.
# 	If it does not, the SQ value is used. For all homes the RF model is used
# 	to predict the 'mainheat-description'. The results are then used to calculate
# 	the additional peak load on the electrical network if the homes without
# 	electric heating are converted to electric heating. A seperate file is
# 	used to aggregate those values and compare them to the currently network
# 	capacity.
# 	
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

import modelfunction as mf

import inspect
import gc

import combining_data_and_seperating_epc as CASE
import similarity_quantification_model as SQM
import multiclass_randomforest as RFM
import combining_SQ_and_RandomForest_models as combining_models
import combining_results_for_output as combining_results


DATA_PATH = 'data/processed/'
PLOT_PATH = 'plots/'

'''CONFIG dict storing global information. 
	random_int: integer setting the reandom seed for reproducibility.
'''

CONFIG = {
		'random_int': 123
		'input_features':[]					}

# setting random seed for reporducibility. SK learn uses numpy random seed.
np.random.seed(CONFIG['random_int'])



def calculating_epc_accuracy(y_test, y_pred, metrics, name):

	order = ['total', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
	if metrics.empty:
		metrics = pd.DataFrame({'order':order})
		metrics.set_index('order', inplace=True, drop=True)
	
	acc_score = []

	results = pd.DataFrame({'true': y_test,
						'pred': y_pred})

	A = results[results['true']=='A']
	B = results[results['true']=='B']
	C = results[results['true']=='C']
	D = results[results['true']=='D']
	E = results[results['true']=='E']
	F = results[results['true']=='F']
	G = results[results['true']=='G']

	acc_score.append(accuracy_score(results['true'], results['pred']))
	acc_score.append(accuracy_score(A['true'].to_numpy(), A['pred'].to_numpy()))
	acc_score.append(accuracy_score(B['true'].to_numpy(), B['pred'].to_numpy()))
	acc_score.append(accuracy_score(C['true'].to_numpy(), C['pred'].to_numpy()))
	acc_score.append(accuracy_score(D['true'].to_numpy(), D['pred'].to_numpy()))
	acc_score.append(accuracy_score(E['true'].to_numpy(), E['pred'].to_numpy()))
	acc_score.append(accuracy_score(F['true'].to_numpy(), F['pred'].to_numpy()))
	acc_score.append(accuracy_score(G['true'].to_numpy(), G['pred'].to_numpy()))

	metrics[name] = acc_score

	return metrics


def main():

	print('Combining and seperating the initial data....')
	proxies, epc = CASE.main()

	X_train, X_test = train_test_split(epc, test_size=0.3, random_state=CONFIG['random_int'])
	y_test = X_test['current-energy-rating']

	print('Similarity Quantification Model....')
	SQ_results = SQM.main(epc_df=X_train, all_df=X_test)

	metrics = calculating_epc_accuracy(y_test, SQ_results['SQ_current-energy-rating'], pd.DataFrame(), 'SQ')

	print('Training the EPC Random Forest Model and predicting on homes with similarities....')
	RF_SQ_results = RFM.main(X_train, SQ_results, target='current-energy-rating', predicting='epc', saved_file_name='epc_predictions_homes_with_sim', to_fit=True, file_path=None)

	metrics = calculating_epc_accuracy(y_test, RF_SQ_results['RF_current-energy-rating'], metrics, 'RF')

	# print('Predicting the EPC Random Forest Model and predicting on homes without similarities....')
	# no_sim = RFM.main(train_df=pd.DataFrame(), test_df=no_sim, target='current-energy-rating', predicting='epc', saved_file_name='epc_predictions_homes_without_sim', to_fit=False, file_path=None)

	print('Comparing and combining the SQ and RF models....')
	combined_df = combining_models.main(RF_SQ_results)

	metrics = calculating_epc_accuracy(y_test, combined_df['current-energy-rating'], metrics, 'ensamble')

	metrics.to_csv('outputs/testing_metrics_df.csv')
	combined_df.to_csv('outputs/testing_combined_df.csv')

	print(full_dataset.isnull().sum())






if __name__ == '__main__':

	main()		# running the main script

	print('It ran. Good job!')





















