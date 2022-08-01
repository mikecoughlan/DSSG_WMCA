##############################################################################
#
#
#	DSSG_WMCA/scripts/models/modelfunction.py
#
#	Collections of functions that are common to many model scripts so 
#	I don't have to keep re-writing them.
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

import inspect
import gc

DATA_PATH = 'data/processed/'
PLOT_PATH = 'plots/'


def splitting_train_and_test(df, target, test_size, change_non_numeric=False):
	'''Splits the sequentially saved csv file into the training and testing set. 
		File had testing set concat on the training set, so sequentioal split is performed.

	INPUTS:
		df (pd.df): processed data frame with the bottom (test_size %) being the test set.
		target (str): columns for the target array.
		test_size (float between 0 and 1): portion of the dataset that will be segmented for testing.

	RETURNS:
		X_train (np.array): input training array for fitting the model.
		X_test (np.array): input testing array to predict the fit model on.
		y_train (np.array): target array for fitting the model.
		y_test (np.array): ground truth for comparing the models predictions.'''

	if change_non_numeric:

		y_train = target[:int((len(df)*(1-test_size)))]
		y_test = target[int((len(df)*(1-test_size))):]

		y_train = to_numeric(y_train)
		y_test = to_numeric(y_test)
	else:
		y_train = target[:int((len(df)*(1-test_size)))].to_numpy()
		y_test = target[int((len(df)*(1-test_size))):].to_numpy()

	X_train = df[:int((len(df)*(1-test_size)))]
	X_test = df[int((len(df)*(1-test_size))):]

	train_LL = X_train[['uprn', 'LATITUDE', 'LONGITUDE']]
	test_LL = X_test[['uprn', 'LATITUDE', 'LONGITUDE']]

	X_train = X_train.to_numpy()
	X_test = X_test.to_numpy()


	return X_train, X_test, y_train, y_test, train_LL, test_LL


def creating_scaler_for_NN(X_train, X_test):
	'''Fits and trasforms a scaler on the training set and then transforms the 
		test set using the fit scaler.'''

	scaler = StandardScaler()									# defining the type of scaler to use
	print('Fitting scaler....')
	scaler.fit(X_train)									# fitting the scaler to the longest storm
	print('Scaling training data....')
	X_train = scaler.transform(X_train)		# doing a scaler transform to each storm individually
	X_test = scaler.transform(X_test)		# doing a scaler transform to each storm individually
	# n_features = X_train[1].shape[1]

	return X_train, X_test, scaler

def to_numeric(series):
	'''Taking the A-G letter system for the EPC ratings and turning it into numeric for the models to classify'''

	series.replace({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6}, inplace=True)
	series=series.to_numpy()

	return series


def calculating_heat_metrics(test, y_pred, model_names):

	metrics = pd.DataFrame({'models':model_names})
		
	non_elec_acc, elec_acc = [], []
	non_elec_rmse, elec_rmse = [], []
	PR, ROC = [], []
	macro_F1 = []
	
	for y_test, pred, name in zip(test, y_pred, model_names):

		print(name)
		results = pd.DataFrame({'y_test': y_test,
							'pred_non_elec': pred[:,0],
							'pred_elec':pred[:,1]})

		elec = results[results['y_test']==1]
		non_elec = results[results['y_test']==0]

		pred_binary = np.argmax(pred, axis=1)

		binary_results = pd.DataFrame({'y_test': y_test,
				'binary_pred':pred_binary})

		binary_elec = binary_results[binary_results['y_test']==1]
		binary_non_elec = binary_results[binary_results['y_test']==0]

		non_elec_acc.append(accuracy_score(binary_non_elec['y_test'].to_numpy(), binary_non_elec['binary_pred'].to_numpy()))
		elec_acc.append(accuracy_score(binary_elec['y_test'].to_numpy(), binary_elec['binary_pred'].to_numpy()))


		non_elec_rmse.append(np.sqrt(mean_squared_error(non_elec['y_test'].to_numpy(), non_elec['pred_elec'].to_numpy())))
		elec_rmse.append(np.sqrt(mean_squared_error(elec['y_test'].to_numpy(), elec['pred_elec'].to_numpy())))

		prec, rec, ____ = precision_recall_curve(y_test, pred[:,1])	# calling the precision recall function which outputs arrays of the precision, recall and threshold data
		PR.append(auc(rec, prec))

		roc_score = roc_auc_score(y_test, pred[:,1])

		ROC.append(roc_score)

		f1 = f1_score(y_test, pred_binary, average='macro')

		macro_F1.append(f1)


	metrics['non_elec_accuracy'] = non_elec_acc
	metrics['elec_accuracy'] = elec_acc
	metrics['non_elec_rmse'] = non_elec_rmse
	metrics['elec_rmse'] = elec_rmse
	metrics['PR'] = PR
	metrics['roc_score'] = ROC
	metrics['F1_macro'] = macro_F1

	metrics.set_index('models', inplace=True)


	return metrics



def calculating_epc_metrics(y_test, y_pred, model_names):

	metrics = pd.DataFrame({'models':model_names})
	test_encoded = pd.get_dummies(y_test)
	y_test = test_encoded.to_numpy()
	y_test = np.argmax(y_test, axis=1)
	
	total_acc, A_acc, B_acc, C_acc, D_acc, E_acc, F_acc, G_acc = [], [], [], [], [], [], [], []
	A_rmse, B_rmse, C_rmse, D_rmse, E_rmse, F_rmse, G_rmse = [], [], [], [], [], [], []
	ROC, macro_F1 = [], []

	for pred, name in zip(y_pred, model_names):

		print(name)

		results = pd.DataFrame({'y_test': y_test,
							'A':pred[:,0],
							'B':pred[:,1],
							'C':pred[:,2],
							'D':pred[:,3],
							'E':pred[:,4],
							'F':pred[:,5],
							'G':pred[:,6]})

		A_prob = results[results['y_test']==0]
		B_prob = results[results['y_test']==1]
		C_prob = results[results['y_test']==2]
		D_prob = results[results['y_test']==3]
		E_prob = results[results['y_test']==4]
		F_prob = results[results['y_test']==5]
		G_prob = results[results['y_test']==6]

		A_prob['y_test'] = 1
		B_prob['y_test'] = 1
		C_prob['y_test'] = 1
		D_prob['y_test'] = 1
		E_prob['y_test'] = 1
		F_prob['y_test'] = 1
		G_prob['y_test'] = 1

		

		pred_binary = np.argmax(pred, axis=1)

		binary_results = pd.DataFrame({'y_test': y_test,
				'binary_pred':pred_binary})

		A = binary_results[binary_results['y_test']==0]
		B = binary_results[binary_results['y_test']==1]
		C = binary_results[binary_results['y_test']==2]
		D = binary_results[binary_results['y_test']==3]
		E = binary_results[binary_results['y_test']==4]
		F = binary_results[binary_results['y_test']==5]
		G = binary_results[binary_results['y_test']==6]


		total_acc.append(accuracy_score(binary_results['y_test'].to_numpy(), binary_results['binary_pred'].to_numpy()))
		A_acc.append(accuracy_score(A['y_test'].to_numpy(), A['binary_pred'].to_numpy()))
		B_acc.append(accuracy_score(B['y_test'].to_numpy(), B['binary_pred'].to_numpy()))
		C_acc.append(accuracy_score(C['y_test'].to_numpy(), C['binary_pred'].to_numpy()))
		D_acc.append(accuracy_score(D['y_test'].to_numpy(), D['binary_pred'].to_numpy()))
		E_acc.append(accuracy_score(E['y_test'].to_numpy(), E['binary_pred'].to_numpy()))
		F_acc.append(accuracy_score(F['y_test'].to_numpy(), F['binary_pred'].to_numpy()))
		G_acc.append(accuracy_score(G['y_test'].to_numpy(), G['binary_pred'].to_numpy()))

		A_rmse.append(np.sqrt(mean_squared_error(A_prob['y_test'].to_numpy(), A_prob['A'].to_numpy())))
		B_rmse.append(np.sqrt(mean_squared_error(B_prob['y_test'].to_numpy(), B_prob['B'].to_numpy())))
		C_rmse.append(np.sqrt(mean_squared_error(C_prob['y_test'].to_numpy(), C_prob['C'].to_numpy())))
		D_rmse.append(np.sqrt(mean_squared_error(D_prob['y_test'].to_numpy(), D_prob['D'].to_numpy())))
		E_rmse.append(np.sqrt(mean_squared_error(E_prob['y_test'].to_numpy(), E_prob['E'].to_numpy())))
		F_rmse.append(np.sqrt(mean_squared_error(F_prob['y_test'].to_numpy(), F_prob['F'].to_numpy())))
		G_rmse.append(np.sqrt(mean_squared_error(G_prob['y_test'].to_numpy(), G_prob['G'].to_numpy())))


		roc_score = roc_auc_score(y_test, pred, multi_class='ovo')

		ROC.append(roc_score)

		f1 = f1_score(y_test, pred_binary, average='macro')

		macro_F1.append(f1)

	metrics['total_accuracy'] = total_acc
	metrics['A_accuracy'] = A_acc
	metrics['B_accuracy'] = B_acc
	metrics['C_accuracy'] = C_acc
	metrics['D_accuracy'] = D_acc
	metrics['E_accuracy'] = E_acc
	metrics['F_accuracy'] = F_acc
	metrics['G_accuracy'] = G_acc
	metrics['A_rmse'] = A_rmse
	metrics['B_rmse'] = B_rmse
	metrics['C_rmse'] = C_rmse
	metrics['D_rmse'] = D_rmse
	metrics['E_rmse'] = E_rmse
	metrics['F_rmse'] = F_rmse
	metrics['G_rmse'] = G_rmse
	metrics['roc_score'] = ROC
	metrics['F1_macro'] = macro_F1

	metrics.set_index('models', inplace=True)

	return metrics


def scatter_plot(y_true, y_pred):

	fig = plt.figure()

	plt.scatter(y_true, y_pred[:,1])
	plt.savefig(PLOT_PATH+'KNN_scatterplot.png')















