##################################################################################################
#
# DSSG_WMCA/scripts/models/naive_KNN.py
#
# Runs a naive k-nearest-neighbors model using just latitude and longitude to determine the EPC
# rating and the binary mainheat-description of a property.
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import precision_recall_curve, f1_score, auc, roc_curve, confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from tensorflow.keras.utils import to_categorical

import modelfunction as mf

import inspect
import gc

DATA_PATH = 'data/processed/'
PLOT_PATH = 'plots/'

CONFIG = {'n_estimators':10,
		'random_int': 123,
		'test_size': 0.3,
		'target': 'mainheat-description'
		}

# setting random seed for reporducibility. SK learn uses numpy random seed.
np.random.seed(CONFIG['random_int'])

def KNN(X_train, X_test, y_train, y_test):

	neighbors = [i for i in range(1, 101)]
	roc_auc, elec_auc, non_elec_auc = [], [], []
	accuracy, elec_accuracy, non_elec_accuracy = [], [], []
	A_accuracy, B_accuracy, C_accuracy, D_accuracy = [], [], [], []
	E_accuracy, F_accuracy, G_accuracy = [], [], []

	# y_train = to_categorical(y_train.to_numpy(), num_classes=7)
	y_test=np.argmax(y_test, axis=1)

	# for i in tqdm(range(1,101)):
	model = KNeighborsClassifier(n_neighbors=5, weights='distance')
	model.fit(X_train, y_train)
	y_pred = model.predict_proba(X_test)
	row_sums = np.sum(y_pred, axis=1) # normalization 
	row_sums = np.reshape(row_sums, (len(row_sums),1))
	copy_array = row_sums
	row_sums = np.pad(row_sums,((0,0),(0,1)), mode="edge")
	y_pred = np.divide(y_pred, row_sums) # these should be histograms
	y_pred_save = pd.DataFrame(y_pred)
	y_pred_save.to_csv('outputs/mainheat/knn_naive_model.csv', index=False)
	# y_pred = np.array(y_pred)
	# 	# y_pred = np.squeeze(y_pred, axis=2)
	# 	# y_pred = y_pred.T

	# 	# roc_area = mf.calculating_metrics(y_test, y_pred)

	# 	# roc_auc.append(roc_area)
	# 	# elec_auc.append(area_elec)
	# 	# non_elec_auc.append(area_non_elec)

	# 	# matrix = confusion_matrix(y_test, y_pred)
	# 	# print("confusion matrix: "+str(matrix))

	# 	y_pred=np.argmax(y_pred, axis=1)


	# 	results = pd.DataFrame({'y_test': y_test,
	# 							'y_pred': y_pred})

	# 	# A = results[results['y_test']==0]
	# 	# B = results[results['y_test']==1]
	# 	# C = results[results['y_test']==2]
	# 	# D = results[results['y_test']==3]
	# 	# E = results[results['y_test']==4]
	# 	# F = results[results['y_test']==5]
	# 	# G = results[results['y_test']==6]
	# 	elec = results[results['y_test']==1]
	# 	non_elec = results[results['y_test']==0]

	# 	elec_acc = accuracy_score(elec['y_test'], elec['y_pred'])
	# 	elec_accuracy.append(elec_acc)
	# 	print('KNN elec Accuracy Score: '+str(elec_acc))

	# 	non_elec_acc = accuracy_score(non_elec['y_test'], non_elec['y_pred'])
	# 	non_elec_accuracy.append(non_elec_acc)
	# 	print('KNN non elec Accuracy Score: '+str(non_elec_acc))

	# 	# A_acc = accuracy_score(A['y_test'], A['y_pred'])
	# 	# A_accuracy.append(A_acc)
	# 	# print('KNN A Accuracy Score: '+str(A_acc))

	# 	# B_acc = accuracy_score(B['y_test'], B['y_pred'])
	# 	# B_accuracy.append(B_acc)
	# 	# print('KNN B Accuracy Score: '+str(B_acc))

	# 	# C_acc = accuracy_score(C['y_test'], C['y_pred'])
	# 	# C_accuracy.append(C_acc)
	# 	# print('KNN C Accuracy Score: '+str(C_acc))

	# 	# D_acc = accuracy_score(D['y_test'], D['y_pred'])
	# 	# D_accuracy.append(D_acc)
	# 	# print('KNN D Accuracy Score: '+str(D_acc))

	# 	# E_acc = accuracy_score(E['y_test'], E['y_pred'])
	# 	# E_accuracy.append(E_acc)
	# 	# print('KNN E Accuracy Score: '+str(E_acc))

	# 	# F_acc = accuracy_score(F['y_test'], F['y_pred'])
	# 	# F_accuracy.append(F_acc)
	# 	# print('KNN F Accuracy Score: '+str(F_acc))

	# 	# G_acc = accuracy_score(G['y_test'], G['y_pred'])
	# 	# G_accuracy.append(G_acc)
	# 	# print('KNN G Accuracy Score: '+str(G_acc))

	# 	acc = accuracy_score(y_test, y_pred)
	# 	accuracy.append(acc)
	# 	print('KNN LAT LON Accuracy Score: '+str(acc))

	# # scores = pd.DataFrame({'neighbors':neighbors,
	# # 						'ROC': roc_auc,
	# # 						'elec_auc':elec_auc,
	# # 						'non_elec_auc':non_elec_auc})

	# # scores = pd.DataFrame({'accuracy':accuracy,
	# # 						'A_acc':A_accuracy,
	# # 						'B_acc':B_accuracy,
	# # 						'C_acc':C_accuracy,
	# # 						'D_acc':D_accuracy,
	# # 						'E_acc':E_accuracy,
	# # 						'F_acc':F_accuracy,
	# # 						'G_acc':G_accuracy})

	# scores = pd.DataFrame({'accuracy':accuracy,
	# 						'elec_acc':elec_accuracy,
	# 						'non_elec_acc':non_elec_accuracy})

	# scores.to_csv('outputs/KNN_scores_mainheat_divided_SMOTE.csv')


	# # fig = plt.figure()
	# # plt.plot(scores['accuracy'], label='total accuracy')
	# # plt.plot(scores['A_acc'], label='A accuracy')
	# # plt.plot(scores['B_acc'], label='B accuracy')
	# # plt.plot(scores['C_acc'], label='C accuracy')
	# # plt.plot(scores['D_acc'], label='D accuracy')
	# # plt.plot(scores['E_acc'], label='E accuracy')
	# # plt.plot(scores['F_acc'], label='F accuracy')
	# # plt.plot(scores['G_acc'], label='G accuracy')
	# # plt.legend()
	# # plt.xlabel('Neighbors')
	# # plt.ylabel('Acc score')
	# # plt.savefig('plots/EPC_KNN_scores.png')

	# fig = plt.figure()
	# plt.plot(scores['accuracy'], label='total accuracy')
	# plt.plot(scores['elec_acc'], label='Electric accuracy')
	# plt.plot(scores['non_elec_acc'], label='Non Electric accuracy')
	# plt.title('KNN Accuracy Scores')
	# plt.legend()
	# plt.xlabel('Neighbors')
	# plt.ylabel('Acc score')
	# plt.savefig('plots/mainheat_KNN_scores.png')

	return model, y_pred


def main(config):

	df = pd.read_csv(DATA_PATH+'numerical_individual_columns_data.csv')

	target = df[config['target']]

	trimmed_cols = ['LONGITUDE', 'LATITUDE']

	df = df[trimmed_cols]


	X_train, X_test, y_train, y_test = mf.splitting_train_and_test(df, target, config['test_size'])

	# over = SMOTE()
	# under = RandomUnderSampler()
	# steps = [('o', over), ('u', under)]
	# pipeline = Pipeline(steps=steps)
	# # transform the dataset
	# X_train, y_train = pipeline.fit_resample(X_train, y_train)

	# train_encoded = pd.get_dummies(y_train)
	# y_train = train_encoded.to_numpy()


	test_encoded = pd.get_dummies(y_test)
	y_test = test_encoded.to_numpy()


	model, y_pred = KNN(X_train, X_test, y_train, y_test)
	mf.scatter_plot(y_test, y_pred)



if __name__ == '__main__':

	main(CONFIG)

	print('It ran. Good job!')