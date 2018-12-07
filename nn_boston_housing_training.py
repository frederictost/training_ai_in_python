# OS functions
import os
import time
# NumPy
import numpy as np
# Pandas
import pandas as pd
# Matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
# TensorFlow
import tensorflow as tf
# Sklearn
from sklearn import preprocessing
import tc_mlp_neural_network as ega_nn
from sklearn.metrics import mean_squared_error

epsilon = 1e-3
np.random.seed(1234)
tf.set_random_seed(1234)

# ---------------------
# 1. Load data
# ---------------------

# Load data
start_time = time.time()
print ("Reading data...")
df = pd.read_table("./data/boston_housing.csv", sep =";", skiprows = [1])
print ("Data was loaded in %f s"%(time.time()-start_time))

# ---------------------
# 2. Shape dataset
# ---------------------
# input_param_list  = ["housingMedianAge","totalRooms","totalBedrooms","population","households","medianIncome"]
input_param_list  = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
output_param_list = ["MEDV"]

X_all = df[input_param_list].values
Y_all = df[output_param_list].values

# Scale input
scaler = preprocessing.StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

scalerY = preprocessing.StandardScaler()
# Y_all_scaled = scaler.fit_transform(Y_all)
Y_all_scaled = Y_all

# plt.scatter(np.arange(len(Y_all_scaled)), Y_all_scaled, s=1, c='r')
# plt.show()

model_name = "model_4_3_2_b10_standard_scaler_norm"

# Init surrogate model
nn = ega_nn.ega_mlp_neural_network(model_name, len(input_param_list), 4, 3, 2, 0.0001, 10, scaler)
training_X, validation_X, training_Y, validation_Y = nn.prepare_dataset(X_all_scaled, Y_all_scaled, ratio = 0.20)
nn.start(True)
# nn.run(training_X, validation_X, training_Y , validation_Y, max_epoch = 2000)

# mach  = df_to_plot_array[:,1]
# alpha = df_to_plot_array[:,2]
# alpha_target = np.ones(len(alpha))*3.5
# xs = df_to_plot_array[:, 3]
# ys = df_to_plot_array[:, 4]
# zs = df_to_plot_array[:, 5]
#

predicted_values = nn.getPredictedValues(X_all_scaled)
delta = (predicted_values - np.ravel(Y_all))
np.savetxt('delta.csv', np.transpose(predicted_values), delimiter=',')

mse = mean_squared_error(Y_all, predicted_values)
print ("MSE %.2f" % mse)

#
# fig = plt.figure(figsize=(11, 8))
# plt.scatter(xs, cp_ref_values, s=1, c='red')
# plt.scatter(xs, cp_predicted_values, s=1, c='blue')
# plt.gca().invert_yaxis()
# plt.show()

                     #"CRIM",  "ZN", " INDUS", "CHAS",  "NOX",  "RM",   "AGE", "DIS",   "RAD",    "TAX",  "PTRATIO",   "B",  "LSTAT"
X_new = np.array( [ [  0.005,    18,    5,        0,      0.5,    3.0,    40.0,  10.0,    3.0,     200,      15.0,      20.0,   5.0 ],
                    [  0.005,    18,    5 ,       1,      0.5,    3.0,    40.0,  10.0,    3.0,     200,      15.0,      20.0,   5.0 ],
                    [ 60,        18,    5,        0,      0.5,    3.0,    40.0,  10.0,    3.0,     200,      15.0,      20.0,   5.0 ],
                    [ 60,        18,    5,        1,      0.5,    3.0,    40.0,  10.0,    3.0,     200,      15.0,      20.0,   5.0] ] )

X_new_scaled = scaler.transform(X_new)
predicted_values = nn.getPredictedValues(X_new_scaled)

print (predicted_values)

nn.closeSession()

