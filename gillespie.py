import numpy as np
from random import random
from operator import add
import math
import functools
import matplotlib.pyplot as plt

if __name__ == "__main__":
    numberOfRepeats = 30
    numberOfTimeSteps = 100
    x0 = np.zeros((numberOfRepeats, 1))
    kb = 20
    kd = 1
    time = np.zeros((numberOfRepeats, numberOfTimeSteps))
    x = np.zeros((numberOfRepeats, numberOfTimeSteps))
    x[:, 0] = x0[:, 0]
    v = np.array([-1, 1])
    for t in list(range(0, numberOfTimeSteps))[1:]:
        propensities = np.array([[kd*x[i, t-1], kb] for i in range(0, numberOfRepeats)])
        totPropensity = np.array([[functools.reduce(add, propensities[i])] for i in range(0, numberOfRepeats)])
        dt = np.array([[math.log(1/random())/totPropensity[i]] for i in range(0, numberOfRepeats)])
        time[:, t] = np.array(list(map(add, time[:, t-1], dt[:, 0])))[:,0]
        culprop = [[0, propensities[i][0], propensities[i][0]+propensities[i][1]] for i in range(0, numberOfRepeats)]
        ra = [random()*totPropensity[i][0] for i in range(0, numberOfRepeats)]
        reac = [[j for j in range(0, len(culprop[i])) if culprop[i][j]-ra[i] < 0][-1] for i in range(0,numberOfRepeats)]
        reac = [v[i] for i in reac]
        x[:, t] = np.array(list(map(add, x[:, t-1], reac)))

    for i in range(0, numberOfRepeats):
        plt.plot(time[i, :], x[i, :])
    plt.show()

