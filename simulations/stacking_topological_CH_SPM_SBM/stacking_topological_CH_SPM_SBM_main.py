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
column_names = mat_vars.pop('column_names')
column_names = [column_names[0][i][0] for i in range(len(column_names[0]))]
scores = mat_vars.pop('scores')
scores_train = mat_vars.pop('scores_train')
del mat_vars

# compute topological features for training and test sets
df_train = compute_topological_features(A_train, edges_train)
df_train = pd.merge(df_train, pd.DataFrame(np.concatenate((edges_train, scores_train, labels_train), axis=1), columns=['i','j']+column_names+['TP']), on=['i', 'j'], sort=False)
df_test = compute_topological_features(A, edges_test)
df_test = pd.merge(df_test, pd.DataFrame(np.concatenate((edges_test, scores), axis=1), columns=['i','j']+column_names), on=['i', 'j'], sort=False)

# link prediction
scores = compute_link_scores(df_train, df_test)

# save Matlab variable
scores[:,0:2] += 1
spio.savemat(fileoutput, {'scores':scores})
