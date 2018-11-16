"""Create surrogate models from data in the folder dtlzdatasets."""

# %% imports
import numpy as np
import pandas as pd
import pickle
from sklearn import svm, neighbors, ensemble
from sklearn.linear_model import SGDRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import r2_score
from os import listdir, getcwd
import warnings
warnings.filterwarnings("ignore")

# %% Actual code
folder = getcwd() + '/dtlzdatasets/'
files = listdir(folder)
outputfolder = folder + 'modellingresults/'
numfiles = len(files)
i = 0
svmparams = {'kernel': ['rbf', 'sigmoid'],
             'gamma': ['auto', 'scale']}
SVMresults = pd.DataFrame(np.zeros((numfiles,4)),
                          columns=('file', 'kernel', 'gamma', 'score'))
nnparams = {'layer': [10, 15, 20],
            'solver': ['lbfgs', 'sgd']}
NNresults = pd.DataFrame(np.zeros((numfiles,4)),
                         columns=('file', 'layer', 'solver', 'score'))
ensembleparams = {'bag': ensemble.BaggingRegressor,
                  'ada': ensemble.AdaBoostRegressor,
                  'GB': ensemble.GradientBoostingRegressor,
                  'RF': ensemble.RandomForestRegressor}
ensembleresults = pd.DataFrame(np.zeros((numfiles,4)),
                         columns=('file', 'type', 'score'))
for file in files:
    print('File', i, 'of', numfiles+1)
    fullfilename = folder + file
    data = pickle.load(open(fullfilename, 'rb'))
    inputs = data[data.columns[0:-2]]
    f1 = data['f1']
    f2 = data['f2']
    inputs_train, inputs_test, f2_train, f2_test = tts(inputs, f2)
    # SVM
    for kernel in svmparams['kernel']:
        for gamma in svmparams['gamma']:
            max_score = 0
            best_model = None
            for j in range(3):
                clf = svm.SVR(gamma=gamma, kernel=kernel)
                clf.fit(inputs_train, f2_train)
                pred = clf.predict(inputs_test)
                score = r2_score(f2_test, pred)
                if score > max_score:
                    max_score = score
                    best_model = clf
            SVMresults['file'][i] = file
            SVMresults['kernel'][i] = kernel
            SVMresults['gamma'][i] = gamma
            SVMresults['score'][i] = score
    # NN
    for layer in nnparams['layer']:
        for solver in nnparams['solver']:
            max_score = 0
            best_model = None
            for j in range(3):
                clf = MLPRegressor(hidden_layer_sizes=layer, solver=solver)
                clf.fit(inputs_train, f2_train)
                pred = clf.predict(inputs_test)
                score = r2_score(f2_test, pred)
                if score > max_score:
                    max_score = score
                    best_model = clf
            NNresults['file'][i] = file
            NNresults['layer'][i] = layer
            NNresults['solver'][i] = solver
            NNresults['score'][i] = score
    # Ensemble
    for type in ensembleparams:
        max_score = 0
        best_model = None
        for j in range(3):
            clf = ensembleparams[type]()
            clf.fit(inputs_train, f2_train)
            pred = clf.predict(inputs_test)
            score = r2_score(f2_test, pred)
            if score > max_score:
                max_score = score
                best_model = clf
        ensembleresults['file'][i] = file
        ensembleresults['type'][i] = type
        ensembleresults['score'][i] = score


# %%
folder = getcwd() + '/dtlzdatasets/'
files = listdir(folder)
data = pickle.load(open((folder+files[0]), "rb"))
inputs = data[data.columns[0:-2]]
f1 = data['f1']
f2 = data['f2']
inputs_train, inputs_test, f2_train, f2_test = tts(inputs, f2)
# %% SVC
clf = svm.SVR(gamma='scale')
clf.fit(inputs_train, f2_train)
pred = clf.predict(inputs_test)
print(r2_score(f2_test, pred))

# %% NNs
clf = MLPRegressor(hidden_layer_sizes=(20, 15,),
                   max_iter=20000,
                   learning_rate='adaptive')
clf.fit(inputs_train, f2_train)
pred = clf.predict(inputs_test)
print(r2_score(f2_test, pred))
#%%
a = pd.DataFrame(np.random.random((2,3)),columns=('a', 'b', 'c'))
a = {'a':1, 'v':2}
for i in a:
    print(i)
#%%
folder = getcwd() + '/dtlzdatasets/'
files = listdir(folder)
outputfolder = folder + 'modellingresults/'
numfiles = len(files)
SVMresults = pd.DataFrame(np.zeros((numfiles, 4)),
                          columns=('file', 'kernel', 'gamma', 'score'))
SVMresults['file'][5] = 1932
SVMresults