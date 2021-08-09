from DataLoader import load_data
from Train import trainCoxPASNet

import torch
import numpy as np

if torch.cuda.is_available():
	print("Using GPU")
else:
	print(" No GPU" * 50)

dtype = torch.FloatTensor
''' Net Settings'''
In_Nodes = 130 ###number of genes
Hidden_Nodes = 100 ###number of hidden nodes
Out_Nodes = 30 ###number of hidden nodes in the last hidden layer
''' Initialize '''
Initial_Learning_Rate = [0.03, 0.01, 0.001, 0.00075]
L2_Lambda = [0.1, 0.01, 0.005, 0.001]
num_epochs = 3000 ###for grid search
Num_EPOCHS = 20000 ###for training
###sub-network setup
Dropout_Rate = [0.7, 0.5]

x_train, ytime_train, yevent_train, age_train = load_data("../data/new_train.csv", dtype)
x_valid, ytime_valid, yevent_valid, age_valid = load_data("../data/new_validation.csv", dtype)
x_test, ytime_test, yevent_test, age_test = load_data("../data/new_test.csv", dtype)


''' Initialize '''
#Initial_Learning_Rate = 0.03
#L2_Lambda = 0.001


def grid_search():
	opt_l2_loss = 0
	opt_lr_loss = 0
	opt_loss = torch.Tensor([float("Inf")])
	###if gpu is being used
	if torch.cuda.is_available():
		opt_loss = opt_loss.cuda()

	###
	opt_c_index_va = 0
	opt_c_index_tr = 0
###grid search the optimal hyperparameters using train and validation data
	for l2 in L2_Lambda:
		for lr in Initial_Learning_Rate:
			loss_train, loss_valid, c_index_tr, c_index_va = trainCoxPASNet(x_train, age_train, ytime_train, yevent_train,
																	x_valid, age_valid, ytime_valid, yevent_valid,
																	In_Nodes, Hidden_Nodes, Out_Nodes,
																	lr, l2, num_epochs, Dropout_Rate)
			if loss_valid < opt_loss:
				opt_l2_loss = l2
				opt_lr_loss = lr
				opt_loss = loss_valid
				opt_c_index_tr = c_index_tr
				opt_c_index_va = c_index_va
			print ("L2: ", l2, "LR: ", lr, "Loss in Validation: ", loss_valid)


	print("Optimal L2: ", opt_l2_loss, "Optimal LR: ", opt_lr_loss)


# grid_search()

### train Cox-PASNet with optimal hyperparameters using train data, and then evaluate the trained model with test data
### Note that test data are only used to evaluate the trained Cox-PASNet

# Optimal L2:  0.1 Optimal LR:  0.00075

Initial_Learning_Rate = 0.00075
L2_Lambda = 0.1

loss_train, loss_test, c_index_tr, c_index_te = trainCoxPASNet(x_train, age_train, ytime_train, yevent_train,
							x_test, age_test, ytime_test, yevent_test,
							In_Nodes, Hidden_Nodes, Out_Nodes,
							Initial_Learning_Rate, L2_Lambda, Num_EPOCHS, Dropout_Rate)

print("C-index in Test: ", c_index_te)
# C-index in Test:  tensor(0.6716, device='cuda:0')
