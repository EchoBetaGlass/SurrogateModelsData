"""Create surrogate models from data in the folder dtlzdatasets."""

# %% imports
import numpy as np
import pandas as pd
import pickle
from sklearn import svm, ensemble
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import r2_score
from os import listdir, getcwd
import warnings

warnings.filterwarnings("ignore")

# %% Actual code
folder = getcwd() + "/dtlzdatasets/"
files = listdir(folder)
outputfolder = folder + "modellingresults/"
numfiles = len(files)
i = 0
svmparams = {"kernel": ["rbf", "sigmoid"], "gamma": ["auto", "scale"]}

SVMresults = pd.DataFrame(columns=("file", "kernel", "gamma", "score"))
SVMresults_temp = pd.DataFrame(
    np.zeros((1, 4)), columns=("file", "kernel", "gamma", "score")
)

nnparams = {"layer": [10, 15, 20], "solver": ["lbfgs", "sgd"]}

NNresults = pd.DataFrame(columns=("file", "layer", "solver", "score"))
NNresults_temp = pd.DataFrame(
    np.zeros((1, 4)), columns=("file", "layer", "solver", "score")
)

ensembleparams = {
    "bag": ensemble.BaggingRegressor,
    "ada": ensemble.AdaBoostRegressor,
    "GB": ensemble.GradientBoostingRegressor,
    "RF": ensemble.RandomForestRegressor,
}
ensembleresults = pd.DataFrame(columns=("file", "type", "score"))
ensembleresults_temp = pd.DataFrame(np.zeros((1, 3)), columns=("file", "type", "score"))
for file in files:
    i = i + 1
    print("File", i, "of", numfiles + 1)
    fullfilename = folder + file
    data = pickle.load(open(fullfilename, "rb"))
    inputs = data[data.columns[0:-2]]
    f1 = data["f1"]
    f2 = data["f2"]
    inputs_train, inputs_test, f2_train, f2_test = tts(inputs, f2)
    # SVM
    for kernel in svmparams["kernel"]:
        for gamma in svmparams["gamma"]:
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
            SVMresults_temp["file"] = file
            SVMresults_temp["kernel"] = kernel
            SVMresults_temp["gamma"] = gamma
            SVMresults_temp["score"] = max_score
            SVMresults = SVMresults.append(SVMresults_temp)
    # NN
    for layer in nnparams["layer"]:
        for solver in nnparams["solver"]:
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
            NNresults_temp["file"] = file
            NNresults_temp["layer"] = layer
            NNresults_temp["solver"] = solver
            NNresults_temp["score"] = max_score
            NNresults = NNresults.append(NNresults_temp)
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
        ensembleresults_temp["file"] = file
        ensembleresults_temp["type"] = type
        ensembleresults_temp["score"] = max_score
        ensembleresults = ensembleresults.append(ensembleresults_temp)
pickle.dump(SVMresults, open("SVMR2", "wb"))
pickle.dump(NNresults, open("NNR2", "wb"))
pickle.dump(ensembleresults, open("ENR2", "wb"))
