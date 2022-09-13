import scipy.io as spio
import numpy as np
import pandas as pd
import sys
from topological_predictors import compute_topological_features
from stacking_link_prediction import compute_link_scores

fileinput = sys.argv[1]
fileoutput = sys.argv[2]

# import Matlab variables
mat_vars = spio.loadmat(fileinput)
A = mat_vars.pop('A')
A = A.toarray()
A_train = mat_vars.pop('A_train')
A_train = A_train.toarray()
edges_train = mat_vars.pop('edges_train')
edges_train = edges_train - 1
edges_test = mat_vars.pop('edges_test')
edges_test = edges_test - 1
labels_train = mat_vars.pop('labels_train')
del mat_vars

# compute topological features for training and test sets
df_train = compute_topological_features(A_train, edges_train)
labels_train = pd.DataFrame(np.concatenate((edges_train, labels_train), axis=1), columns=['i','j','TP'])
df_train = pd.merge(df_train, labels_train, on=['i', 'j'], sort=False)
df_test = compute_topological_features(A, edges_test)

# link prediction
scores = compute_link_scores(df_train, df_test)

# save Matlab variable
scores[:,0:2] += 1
spio.savemat(fileoutput, {'scores':scores})
