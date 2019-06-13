"""
Data visualization
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2017, Stanford University"
__license__ = "MIT"

import os
import deepchem as dc
import numpy as np
import csv
import sklearn
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from binding_pocket_datasets import load_pdbbind_pockets
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

start = time.time()

# For stable runs
#np.random.seed(123)

split = "random"
subset = "refined"
pdbbind_tasks, pdbbind_datasets, transformers = load_pdbbind_pockets(
    split=split, subset=subset)
train_dataset, valid_dataset, test_dataset = pdbbind_datasets
current_dir = os.path.dirname(os.path.realpath(__file__))


# Print error metrics to csv file
def calc_metrics(train_dataset, valid_dataset, transformers, trainrow, validrow):
    for met in ['rms_score', 'mae_score', 'pearson_r2_score', 'r2_score']:
        if met == 'rms_score':
            metric = dc.metrics.Metric(dc.metrics.rms_score)
        elif met == 'mae_score':
            metric = dc.metrics.Metric(dc.metrics.mae_score)
        elif met == 'pearson_r2_score':
            metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
        elif met == 'r2_score':
            metric = dc.metrics.Metric(dc.metrics.r2_score)

        print("Train scores")
        train_scores = model.evaluate(train_dataset, [metric], transformers)
        trainrow.append(train_scores)
        print("Validation scores")
        valid_scores = model.evaluate(valid_dataset, [metric], transformers)
        validrow.append(valid_scores)

    return trainrow, validrow

def calc_pred(test_dataset, transformers, test_row):
    for met in ['rms_score', 'mae_score', 'pearson_r2_score', 'r2_score']:
        if met == 'rms_score':
            metric = dc.metrics.Metric(dc.metrics.rms_score)
        elif met == 'mae_score':
            metric = dc.metrics.Metric(dc.metrics.mae_score)
        elif met == 'pearson_r2_score':
            metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
        elif met == 'r2_score':
            metric = dc.metrics.Metric(dc.metrics.r2_score)
        print("Test scores")
        test_scores = model.evaluate(test_dataset, [metric], transformers)
        test_row.append(test_scores)

    return test_row




# PCA

pca = PCA()
pca.fit(train_dataset.X)
# print(pca.singular_values_)
sv = pca.singular_values_
ex = pca.explained_variance_
ex_ratio = pca.explained_variance_ratio_
# print(sv.shape)

# Feature selection
# tol = 1
# comps = 0
# while (True):
#     err = np.abs(sv[comps + 1] - sv[comps])
#     if (err < tol):
#         break
#     comps += 1
# print('Number of components for tol = 1 is ' + str(comps + 1) + ' out of ' + str(len(sv)))

# Feature selection: Kaiser Rule: eigenvalues of at least 1
# tol = 1
# comps = 0
# while (True):
#     if sv[comps] < 1:
#         break
#     comps += 1
# print('Number of principle components by Kaiser Rule is ' + str(comps + 1) + ' out of ' + str(len(sv)))

# Feature selection: Selected PCs explain at least 0.1% of the variance
tol = 1/100
comps = 0
while (True):
    if ex_ratio[comps] < tol:
        break
    comps += 1
print('Number of principle components that explain at least 0.1% of variance is ' + str(comps + 1) + ' out of ' + str(len(sv)))


# Make Scree Plot

# plt.scatter(np.arange(ex_ratio.shape[0]), ex_ratio)
# plt.axvline(x=comps, c='r')
# plt.title('Scree Plot, PCA on Core Dataset')
# plt.xlabel('Number of PCs')
# plt.ylabel('Singular Values')
# plt_dir = os.path.join(current_dir, "pca_scree_%s_%s" % (subset, split))
# plt.savefig(plt_dir)




# Train, validate, test, and graph on reduced dimensions

pca = PCA(n_components=comps)
pca.fit(train_dataset.X)

type = "train"
Xtrans = pca.transform(train_dataset.X)
data_dir = os.path.join(current_dir, "%s_pockets_trans_%s" % (subset, type))
train_dataset_trans = dc.data.DiskDataset.from_numpy(Xtrans, train_dataset.y, train_dataset.w, train_dataset.ids, data_dir=data_dir)

type = "valid"
Xtrans = pca.transform(valid_dataset.X)
data_dir = os.path.join(current_dir, "%s_pockets_trans_%s" % (subset, type))
valid_dataset_trans = dc.data.DiskDataset.from_numpy(Xtrans, valid_dataset.y, valid_dataset.w, valid_dataset.ids, data_dir=data_dir)

type = "test"
Xtrans = pca.transform(test_dataset.X)
data_dir = os.path.join(current_dir, "%s_pockets_trans_%s" % (subset, type))
test_dataset_trans = dc.data.DiskDataset.from_numpy(Xtrans, test_dataset.y, test_dataset.w, test_dataset.ids, data_dir=data_dir)



with open(subset + '_metrics_pred_pca.csv', mode='w') as model_metrics:
    writer = csv.writer(model_metrics, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Model', 'Data', 'RMS', 'MAE', 'Pearson R2', 'R2'])

    # Random Forests

    for n_est in [50]: #[10, 20, 50, 100, 200, 500, 1000]:
        model_type = "RF"
        sklearn_model = sklearn.ensemble.RandomForestRegressor(n_estimators=n_est)
        model_dir = os.path.join(current_dir, "%s_models_pca/pocket_%s_%s_%s_%s" % (subset, split, subset, model_type, n_est))
        model = dc.models.SklearnModel(sklearn_model, model_dir=model_dir)

        # Fit trained model
        # print("Fitting model on train dataset")
        model.fit(train_dataset_trans)
        model.save()
        trainrow = ['RF ' + str(n_est), 'Train']
        validrow = ['', 'Valid']

        print("Evaluating model " + 'RF' + str(n_est))
        trainrow, validrow = calc_metrics(train_dataset_trans, valid_dataset_trans, transformers, trainrow, validrow)
        writer.writerow(trainrow)
        writer.writerow(validrow)

        # Make predictions
        testrow = ['', 'Test']
        testrow = calc_pred(test_dataset_trans, transformers, testrow)
        writer.writerow(testrow)

        # Generate parity plots
        for dataset in [train_dataset_trans, valid_dataset_trans, test_dataset_trans]:
            type = "Training: "
            if dataset == valid_dataset_trans:
                type = "Validation: "
            if dataset == test_dataset_trans:
                type = "Testing: "

            y_test = model.predict(dataset)

            plt.clf()
            plt.scatter(dataset.y, y_test, color='blue')
            plt.title('%s_PCA_%s_%s_%s_%s' % (type, split, subset, model_type, n_est))
            plt.xlabel('Experimental')
            plt.ylabel('Predictions')
            plt_dir = os.path.join(current_dir, "plots_pca_%s/pocket_%s_%s_%s_%s_%s" % (subset, type, split, subset, model_type, n_est))
            plt.savefig(plt_dir)

    #
    #
    # # Kernel SVMs
    #
    for kernel in ['poly']: #['linear', 'poly', 'rbf', 'sigmoid']:
        model_type = "SVM"
        sklearn_model = sklearn.svm.SVR(kernel=kernel)
        model_dir = os.path.join(current_dir, "%s_models_pca/pocket_%s_%s_%s_%s" % (subset, split, subset, model_type, kernel))
        model = dc.models.SklearnModel(sklearn_model, model_dir=model_dir)

        # Fit trained model
        # print("Fitting model on train dataset")
        model.fit(train_dataset_trans)
        model.save()
        trainrow = ['SVR ' + kernel, 'Train']
        validrow = ['', 'Valid']

        print("Evaluating model " + 'SVR ' + kernel)
        trainrow, validrow = calc_metrics(train_dataset_trans, valid_dataset_trans, transformers, trainrow, validrow)
        writer.writerow(trainrow)
        writer.writerow(validrow)

        # Make predictions
        testrow = ['', 'Test']
        testrow = calc_pred(test_dataset_trans, transformers, testrow)
        writer.writerow(testrow)

        # Generate parity plots
        for dataset in [train_dataset_trans, valid_dataset_trans, test_dataset_trans]:
            type = "Training: "
            if dataset == valid_dataset_trans:
                type = "Validation: "
            if dataset == test_dataset_trans:
                type = "Testing: "

            y_test = model.predict(dataset)

            plt.clf()
            plt.scatter(dataset.y, y_test, color='green')
            plt.title('%s_PCA_%s_%s_%s_%s' % (type, split, subset, model_type, kernel))
            plt.xlabel('Experimental')
            plt.ylabel('Predictions')
            plt_dir = os.path.join(current_dir, "plots_pca_%s/pocket_%s_%s_%s_%s_%s" % (subset, type, split, subset, model_type, kernel))
            plt.savefig(plt_dir)

    # Kernel Ridge Regression

    for kernel in ['laplacian']:  #['linear', 'poly', 'rbf', 'sigmoid', 'laplacian']:
        model_type = "KRR"
        sklearn_model = sklearn.kernel_ridge.KernelRidge(kernel=kernel)
        model_dir = os.path.join(current_dir, "%s_models_pca/pocket_%s_%s_%s_%s" % (subset, split, subset, model_type, kernel))
        model = dc.models.SklearnModel(sklearn_model, model_dir=model_dir)

        # Fit trained model
        # print("Fitting model on train dataset")
        model.fit(train_dataset_trans)
        model.save()
        trainrow = ['KRR ' + kernel, 'Train']
        validrow = ['', 'Valid']

        print("Evaluating model " + 'KRR ' + kernel)
        trainrow, validrow = calc_metrics(train_dataset_trans, valid_dataset_trans, transformers, trainrow, validrow)
        writer.writerow(trainrow)
        writer.writerow(validrow)

        # Make predictions
        testrow = ['', 'Test']
        testrow = calc_pred(test_dataset_trans, transformers, testrow)
        writer.writerow(testrow)

        # Generate parity plots
        for dataset in [train_dataset_trans, valid_dataset_trans, test_dataset_trans]:
            type = "Training: "
            if dataset == valid_dataset_trans:
                type = "Validation: "
            if dataset == test_dataset_trans:
                type = "Testing: "

            y_test = model.predict(dataset)

            plt.clf()
            plt.scatter(dataset.y, y_test, color='red')
            plt.title('%s_PCA_%s_%s_%s_%s' % (type, split, subset, model_type, kernel))
            plt.xlabel('Experimental')
            plt.ylabel('Predictions')
            plt_dir = os.path.join(current_dir, "plots_pca_%s/pocket_%s_%s_%s_%s_%s" % (subset, type, split, subset, model_type, kernel))
            plt.savefig(plt_dir)




tot_time = time.time() - start

print("Compute time = " + str(tot_time) + ' sec')
















#
# np.savetxt("PCA_" + subset, sv, delimiter=',')



# TSNE
# X_compressed = TSNE(n_components=3).fit_transform(xtrans)
#
# fig = plt.figure()
# ax = Axes3D(fig)
#
# ax.scatter(X_compressed[:, 0], X_compressed[:, 1], X_compressed[:, 2])
#
# ax.set_xlabel('Compressed Principal Component X_0')
# ax.set_ylabel('Compressed Principal Component X_1')
# ax.set_zlabel('Compressed Principal Component X_2')
# plt.show()

# # Save data
# npydata = train_dataset.y
# np.savetxt("foo.csv",npydata, delimiter=',')

# print(np.count_nonzero(npydata,axis=1))
