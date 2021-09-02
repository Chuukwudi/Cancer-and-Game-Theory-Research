from DataLoaderCV import load_data
from Train import trainCoxPASNet
from sklearn.model_selection import KFold

import torch
import numpy as np
import pandas as pd

if torch.cuda.is_available():
	print("Using GPU")
else:
	print(" No GPU" * 50)

dtype = torch.FloatTensor
''' Net Settings'''
In_Nodes = 130 ###number of reactions
Hidden_Nodes = 100 ###number of hidden nodes
Out_Nodes = 30 ###number of hidden nodes in the last hidden layer
''' Initialize '''
Initial_Learning_Rate = [0.03, 0.01, 0.001, 0.00075]
L2_Lambda = [0.1, 0.01, 0.005, 0.001]
num_epochs = 3000 ###for grid search
Num_EPOCHS = 20000 ###for training
###sub-network setup
Dropout_Rate = [0.7, 0.5]

# add validation data to training data
train = pd.read_csv("../data/new_train.csv")
validation = pd.read_csv("../data/new_validation.csv")
data = pd.concat([train, validation])

test = pd.read_csv("../data/new_test.csv")

# implement cross validation
seed = 111
kfold = KFold(n_splits=20, shuffle=True, random_state=seed)
# Used kfold instead of stratified kfold because target variable is a prognostic index, PI.

Initial_Learning_Rate = 0.00075
L2_Lambda = 0.1

# cross validation happens
c_indexx = []
for train, valid in kfold.split(data):
	# train the model
	df_train = data.iloc[train, :].copy()
	df_valid = data.iloc[valid, :].copy()
	x_train, ytime_train, yevent_train, age_train = load_data(df_train, dtype)
	x_valid, ytime_valid, yevent_valid, age_valid = load_data(df_valid, dtype)
	loss_train, loss_test, c_index_tr, c_index_te = trainCoxPASNet(x_train, age_train, ytime_train, yevent_train,
																   x_valid, age_valid, ytime_valid, yevent_valid,
																   In_Nodes, Hidden_Nodes, Out_Nodes,
																   Initial_Learning_Rate, L2_Lambda, Num_EPOCHS,
																   Dropout_Rate)

	print("C-index in Test: ", c_index_te)
	c_indexx.append(c_index_te)
# C-index in Test:  tensor(0.6716, device='cuda:0')
# c_indexx = [0.5474, 0.6451, 0.6649, 0.6370, 0.7028,
# 			  0.5179, 0.6458, 0.6101, 0.8158, 0.4730,
# 			  0.6859, 0.5522, 0.5938, 0.7019, 0.5958,
# 			  0.4833, 0.5833, 0.6639, 0.4801, 0.6329]
