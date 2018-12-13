import pickle as pk

from os import getcwd, listdir

import pandas as pd

folder = getcwd() + "/dtlzdatasets/"
files = listdir(folder)
outputfolder = getcwd() + "/csvdatasets/"

for file in files:
    x = pk.load(open(folder+file, 'rb'))
    output = outputfolder + file[0:-2] + '.csv'
    x.to_csv(output, sep=',')