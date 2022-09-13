import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

def prepare_sets(df_train, df_test):

    feature_set = list(df_test.columns)
    feature_set.remove('i')
    feature_set.remove('j')

    X_cvtrain = []
    y_cvtrain = []
    X_cvtest = []
    y_cvtest = []
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for cvtrain_idx, cvtest_idx in skf.split(df_train, df_train.TP):

        X_cvtrain_i = df_train.iloc[cvtrain_idx]
        y_cvtrain_i = X_cvtrain_i['TP']
        X_cvtrain_i = X_cvtrain_i.loc[:, feature_set]
        X_cvtrain_i.fillna(X_cvtrain_i.mean(), inplace=True)
        sm = RandomOverSampler(random_state=42)
        X_cvtrain_i, y_cvtrain_i = sm.fit_sample(X_cvtrain_i, y_cvtrain_i)

        X_cvtest_i = df_train.iloc[cvtest_idx]
        y_cvtest_i = X_cvtest_i['TP']
        X_cvtest_i = X_cvtest_i.loc[:, feature_set]
        X_cvtest_i.fillna(X_cvtest_i.mean(), inplace=True)

        X_cvtrain.append(X_cvtrain_i)
        y_cvtrain.append(y_cvtrain_i)
        X_cvtest.append(X_cvtest_i)
        y_cvtest.append(y_cvtest_i)

    X_train = df_train
    y_train = X_train['TP']
    X_train = X_train.loc[:, feature_set]
    X_train.fillna(X_train.mean(), inplace=True)
    sm = RandomOverSampler(random_state=42)
    X_train, y_train = sm.fit_sample(X_train, y_train)

    X_test = df_test
    X_test = X_test.loc[:, feature_set]
    X_test.fillna(X_test.mean(), inplace=True)

    return X_train, y_train, X_test, X_cvtrain, y_cvtrain, X_cvtest, y_cvtest


def grid_search_crossval(X_cvtrain, y_cvtrain, X_cvtest, y_cvtest, RFC_depth_list, RFC_trees_list):

    fscore_grid = np.zeros((len(RFC_depth_list), len(RFC_trees_list)))
    for i, depth in enumerate(RFC_depth_list):
        for j, trees in enumerate(RFC_trees_list):
            fscore_cv = np.zeros((len(X_cvtrain), 1))
            for k in range(len(X_cvtrain)):
                RFC_model = RandomForestClassifier(max_depth=depth, n_estimators=trees).fit(X_cvtrain[k], y_cvtrain[k])
                RFC_predictions = RFC_model.predict(X_cvtest[k])
                _, _, fscores, _ = precision_recall_fscore_support(y_cvtest[k], RFC_predictions, average=None)
                fscore_cv[k] = fscores[1]
            fscore_grid[i, j] = np.mean(fscore_cv)

    i, j = np.unravel_index(fscore_grid.argmax(), fscore_grid.shape)
    return RFC_depth_list[i], RFC_trees_list[j]


def RFC_link_prediction(X_test, X_train, y_train, RFC_depth, RFC_trees):

    RFC_model = RandomForestClassifier(max_depth=RFC_depth, n_estimators=RFC_trees).fit(X_train, y_train)
    RFC_prob = RFC_model.predict_proba(X_test)
    RFC_prob = RFC_prob[:, 1]
    return RFC_prob


def compute_link_scores(df_train, df_test):

    # prepare final training and cross-validation sets
    X_train, y_train, X_test, X_cvtrain, y_cvtrain, X_cvtest, y_cvtest = prepare_sets(df_train, df_test)

    # grid search and cross-validation for parameter selection of random forest classifier
    RFC_depth_list = [3, 6]
    RFC_trees_list = [25, 50, 100]
    RFC_depth, RFC_trees = grid_search_crossval(X_cvtrain, y_cvtrain, X_cvtest, y_cvtest, RFC_depth_list, RFC_trees_list)

    # link prediction with random forest classifier
    RFC_prob = RFC_link_prediction(X_test, X_train, y_train, RFC_depth, RFC_trees)

    # output
    return np.concatenate((df_test.loc[:,['i','j']], RFC_prob.reshape((len(RFC_prob), 1))), axis=1)
