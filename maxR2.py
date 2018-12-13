import pickle as pk
import warnings
from os import getcwd, listdir

import numpy as np
import pandas as pd

folder = getcwd() + "/dtlzdatasets/"
files = listdir(folder)
outputfolder = folder + "modellingresults/"
numfiles = len(files)
EN = pk.load(open("ENR2", "rb"))
NN = pk.load(open("NNR2", "rb"))
SVM = pk.load(open("SVMR2", "rb"))
R2_def = pk.load(open("R2results.p", "rb"))
R2_max = pd.DataFrame(columns=["file", "SVM", "GPR", "NN", "EN"])
R2_temp = pd.DataFrame(np.zeros((1, 5)), columns=["file", "SVM", "GPR", "NN", "EN"])
i = 0
for file in files:
    R2_temp["EN"] = max(EN[EN["file"] == file]["score"])
    R2_temp["NN"] = max(NN[NN["file"] == file]["score"])
    R2_temp["SVM"] = max(SVM[SVM["file"] == file]["score"])
    R2_temp["GPR"] = R2_def["GPR"][i]
    R2_temp['file'] = file
    i = i + 1
    R2_max = R2_max.append(R2_temp)
pk.dump(R2_max, open("R2_max", "wb"))
