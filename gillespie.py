import numpy as np
from random import random
from operator import add, sub, mul
import math
import functools
import matplotlib.pyplot as plt
from scipy.stats import skew
from time import time as tnow

if __name__ == "__main__":
    numberOfRepeats = 1000
    numberOfTimeSteps = 10000
    x0 = np.zeros((numberOfRepeats, 1))
    kb = 20
    kd = 1
    time = np.zeros((numberOfRepeats, numberOfTimeSteps))
    x = np.zeros((numberOfRepeats, numberOfTimeSteps))
    x[:, 0] = x0[:, 0]
    v = np.array([-1, 1])
    print("Modeling {} Events".format(numberOfRepeats*numberOfTimeSteps))
    tstart = tnow()
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
    deltat = tnow() - tstart
    print("Modeling took {} seconds, at {} models/second".format(deltat, numberOfTimeSteps*numberOfRepeats/deltat))


    # Model Solutions
    plt.figure(1)
    plt.title("Gillespie Model")
    plt.xlabel("Time(s)")
    plt.ylabel("Num of Member")
    avg = []
    var = []
    skw = []
    for i in range(0, numberOfRepeats):
        plt.plot(time[i, :], x[i, :])
    for i in range(0,numberOfTimeSteps):
        avg.append(np.mean(x[:, i]))
        var.append(np.var(x[:, i]))
        skw.append(skew(x[:, i]))

    plt.figure(4)
    plt.title("Averages, variances and skew of model")
    plt.xlabel("Time(s)")
    plt.ylabel("Num of Member")
    plt.plot(avg)
    plt.plot(var)
    plt.plot(skw)

    # Analytic Solutions
    plt.figure(2)
    plt.title("Analytical plot of simple birthdeath")
    plt.xlabel("Time(s)")
    plt.ylabel("Num of Member")
    analAvg = [kb/kd - kb/kd*math.exp(-kd*t) for t in time[0,:]]
    analVar = [kb/kd - kb/kd*math.exp(-kd*t) for t in time[0,:]]
    analSkew = [0 for t in time[0,:]]
    plt.plot(analAvg)
    plt.plot(analVar)
    plt.plot(analSkew)

    # analytic and model difference
    plt.figure(3)
    plt.title("Difference between simple birthdeath model and analytic")
    plt.xlabel("Time(s)")
    plt.ylabel("Num of Member")
    difAvg = list(map(sub,avg,analAvg))
    difVar = list(map(sub,var,analVar))
    difSkw = list(map(sub,skw,analSkew))
    plt.plot(difAvg)
    plt.plot(difVar)
    plt.plot(difSkw)


    # Modeling Autocorrelation
    auto = []
    for i in range(1, numberOfTimeSteps):
        auto.append(np.mean(list(map(mul, x[:, i-1], x[:i])))-np.mean(x[:, i-1])*np.mean(x[:, i]))
    analAuto = [np.var(x[:, i])*math.exp(-kd*time[:, i][0]) for i in range(1,numberOfTimeSteps)]
    plt.figure(5)
    plt.title("Modelled and analytical Autocorrelation")
    plt.xlabel("Time(s)")
    plt.plot(auto)
    plt.plot(analAuto)

    # Ensemble Avg vs Time average
    selec = 0
    timeAvg = np.mean(x[selec,:])
    ensemAvg = np.mean(x[:, -1])
    print("Time Average:{}, Ensemble Average:{}".format(timeAvg, ensemAvg))

    # Gather first pass process data
    thresh = 18
    cross = []
    for mem in range(0,numberOfRepeats):
        for i in range(0,numberOfTimeSteps):
            if x[mem,i] > thresh:
                cross.append(time[mem, i])
                break
    shape,scale = 2.5,kb*math.log(1/random())/totPropensity[-1]
    analcross = np.random.gamma(shape, scale, 1000)
    plt.figure(6)
    plt.title("Modeled Points of crossing threshold {}".format(thresh))
    plt.hist(cross, bins=15)
    plt.figure(7)
    plt.title("Analytical Points of crossing threshold {}".format(thresh))
    plt.hist(analcross, bins=15)


    plt.show()

