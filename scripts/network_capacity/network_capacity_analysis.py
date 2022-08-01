##############################################################################
#
#
#	DSSG_WMCA/scripts/network_capacity_analysis.py
#
#	Takes the results from the EPC and the heating models, puts them together
# 	in one dataframe. Then calculates the amount of additional load that 
# 	would be put on the system if homes were converted to heatpumps
# 	from non-electric sources. Does this by calculating a ratio of the 
# 	mean of the heating cost current for each EPC band, divided by the mean
# 	for all bands, and then multiplying that by the average heating power
# 	used by homes in the UK (15,000 kWh/year)
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
from scipy.stats import pearsonr
from tqdm import tqdm
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import precision_recall_curve, f1_score, auc, roc_curve, confusion_matrix

from tensorflow.keras.utils import to_categorical

import modelfunction as mf
import inspect
import gc

DATA_PATH = 'data/processed/'
PLOT_PATH = 'plots/'

CONFIG = {
		'random_int': 123,
		'test_size': 0.3,
		'rel_features': ['uprn', 'LATITUDE', 'LONGITUDE', 'current-energy-rating', 'mainheat-description']
							}

# setting random seed for reporducibility. SK learn uses numpy random seed.
np.random.seed(CONFIG['random_int'])


def creating_source_column(train_df, test_df):

	'''Creates a binary column indicating whether the heat and epc values for this 
		property were taken from teh EPC database or were predicted. 0 indicates the data
		was taken from the EPC database, and 1 indicates the data was predicted.'''

	print('Creating source column....')
	train_df['predicted'] = 0
	test_df['predicted'] = 1

	return train_df, test_df


def estimating_power_load_by_EPC(EPC_df, avg_heating_power=15000):

	'''	15,000 kWh is the assumed heating energy usage for UK homes by the uk gov 
		(https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/853753/QEP_Q3_2019.pdf)
	'''
	print('Estimating power load by EPC....')
	costs = EPC_df.groupby(['current-energy-rating', 'mainheat-description']).mean()[['heating-cost-current', 'energy-consumption-current']]

	costs['cost-ratio'] = costs['heating-cost-current']/EPC_df['heating-cost-current'].mean()
	costs['heating-energy-usage'] = round(costs['cost-ratio']*avg_heating_power,0) 


	return costs



def calculating_additional_load(merged_df, EPC_df):

	'''calculates the additional load put on the network by switching a property to 
		an electric heating source. For homes currently using an electric heating source, the 
		value will be 0. This assumes a Coefficient of Performance (COP) of 1 for the 
		added heat pump. For a different COP, divide the heating-energy-usage column of the costs_df
		by the desired COP. The calculation is done using teh UK gov, estimated power usage for heating
		in the UK, multiplied by the ratio of the heating cost for a particular EPC band, divided by the mean.'''

	costs_df = estimating_power_load_by_EPC(EPC_df)

	print('Calculating additional load....')

	ratings = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

	conditions = [(merged_df['mainheat-description'] == 0) & (merged_df['current-energy-rating'] == rating) for rating in ratings]

	additional_loads = [costs_df.loc[rating,1]['heating-energy-usage'] for rating in ratings]

	print(conditions)
	print(additional_loads)
	
	merged_df['additional_load'] = np.select(conditions, additional_loads, default=0)
	merged_df['quarterly_peak'] = merged_df['additional_load']*1.5/4
	print(merged_df['additional_load'])

	return merged_df


def maxing_and_mapping(epc_df, heat_df):

	print('Maxing and Mapping....')

	to_map = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G'}

	epc_df = pd.Series(np.argmax(epc_df.to_numpy(), axis=1)).map(to_map)
	heat_df = pd.Series(np.argmax(heat_df.to_numpy(), axis=1))


	return epc_df, heat_df



def main():

	print('Loading dfs....')
	initial_df = pd.read_csv(DATA_PATH+'numerical_individual_columns_data.csv')
	df = initial_df[CONFIG['rel_features']]

	print('Splitting data frames into train and test....')
	X_train = df[:int((len(df)*(1-CONFIG['test_size'])))]
	X_test = df[int((len(df)*(1-CONFIG['test_size']))):].reset_index(drop=True)
	X_test.drop(['current-energy-rating', 'mainheat-description'], axis=1, inplace=True)

	print('Loading results data frames....')
	epc_df = pd.read_csv('outputs/epc/RandomForest_version_1.csv', header=0, names=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
	heat_df = pd.read_csv('outputs/mainheat/RandomForest_version_1.csv')

	epc_df, heat_df = maxing_and_mapping(epc_df, heat_df)

	X_test['current-energy-rating'] = epc_df
	X_test['mainheat-description'] = heat_df

	X_train, X_test = creating_source_column(X_train, X_test)

	merged_df = pd.concat([X_train, X_test], axis=0, ignore_index=True)

	merged_df = calculating_additional_load(merged_df, initial_df)

	print('Additional Total Load (kWh) on the Network from switching to heatpumps with COP=1: '+str(round(merged_df['additional_load'].sum(),0)))
	print('Additional Quarterly Peak Load (kWh) on the Network from switching to heatpumps with COP=1: '+str(round(merged_df['quarterly_peak'].sum(),0)))






if __name__ == '__main__':

	main()

	print('It ran. Good job.')















