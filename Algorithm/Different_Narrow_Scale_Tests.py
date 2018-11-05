# -*- coding: utf-8 -*-
"""
 Multiscale drift detection algorithm

"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from numpy.random import random
from numpy.random import randint
import bisect
import matplotlib.pyplot as plt


def cross_val(clf,data,t):
    """
    split data set into validation set to obtain cv features
    :param clf: optimal classifier
    :param data: batch data
    :param t: current time t
    :return: cv features(accuracy) for current batch
    """
    x,y = data
    cv = 6
    kf = KFold(n_splits=cv, random_state=1)
    feature = np.zeros((cv,2))
    for i,(train,cv_ind) in enumerate(kf.split(x,y)):
        feature[i,1] = clf.score(x[cv_ind],y[cv_ind])
    feature[:,0] = t
    return feature


def initialization(firstbatch,classifier,scaler):
    """
    initialize accuracy list, detection feature list, and a primary classifier
    :param firstbatch: first batch data (batchdata[0])
    :return: a classifier, detection feature D, and accuracy list acc
    """
    # parameter initialization
    D = []
    acc = []

    # classifier initialization
    if classifier != 'neural networks':
        if classifier == 'linear_regression':
            clf = LogisticRegression()
        elif classifier == 'SVM':
            clf = SVC()
        else:
            clf = KNeighborsClassifier()
        clf.fit(firstbatch["xtrain"], firstbatch["ytrain"])
        D.extend(cross_val(clf, data=(firstbatch["xtest"], firstbatch["ytest"]), t=1))
        acc.append(clf.score(firstbatch["xtest"], firstbatch["ytest"]))
    else:
        scaler.fit(firstbatch["xtrain"])
        X_train = scaler.transform(firstbatch["xtrain"])
        clf = MLPClassifier()
        clf.fit(X_train,firstbatch["ytrain"])
        D.extend(cross_val(clf, data=(scaler.transform(firstbatch["xtest"]), firstbatch["ytest"]), t=1))
        acc.append(clf.score(scaler.transform(firstbatch["xtest"]), firstbatch["ytest"]))
    return clf, D, acc


def weighted_sampling(S_telta,windowsize):
    """
    (optional) weighted sampling to obtain from S_telta to S
    :param S_telta: S_telta
    :param windowsize: number of features in S
    :return: a weighted sampling sequence of index
    """
    totals = []
    running_total = 0
    weights = np.squeeze(np.array(range(S_telta)))
    for w in weights:
        running_total += w
        totals.append(running_total)

    rand_ind = []
    for j in range(windowsize):
        rnd = random()*totals[-1]
        rand_ind.append(bisect.bisect_right(totals,rnd))
    return np.squeeze(rand_ind)


def TS_split(D,windowsize):
    """
    split D into T,S . S is sampled from S_telta. when the result of broad_scale is true,
    check in detail the last window before T (T_minus)
    :param D:
    :param windowsize:
    :return: current test window T, last test window T_minus, sub-stationary window S
    """
    dim = D[0].shape[0]
    T = np.array(D[-windowsize:]).reshape((windowsize,dim))
    S_telta = np.array(D[:-windowsize]).reshape((len(D)-windowsize,dim))
    T_minus = S_telta[-windowsize:]

    # select index for S

    rand_ind = randint(0, len(S_telta), windowsize)
    # rand_ind = weighted_sampling(S_telta.shape[0],windowsize)
    S = S_telta[rand_ind,:]
    return T, T_minus, S


def DD_broad_scale(T,S):
    """
    examine if T and S are significantly different
    :param T: the test window
    :param S: the sub-stationary window
    :return: check result
    """
    s_mean = np.mean(S[:,-1])
    t_mean = np.mean(T[:,-1])
    s_var = np.var(S[:,-1])
    t_var = np.var(T[:,-1])
    delta = 0.025
    # 0.05 - quantile for normal distribution
    t_alpha = 1.65
    Sw = (s_var + t_var) / 2.0
    t = (np.abs(s_mean - t_mean) - delta) / (Sw * np.sqrt(2.0 / S.shape[0]))
    # print "broadscale",t
    if t > t_alpha:
        result = True
        # print result
    else:
        result = False
    return result


def DD_narrow_scale(S):
    """

    :param S:
    :return:
    """
    F_alpha = 1.24
    beta = 1-1.0/(F_alpha+1)
    betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    m = int(np.floor(betas[8] * S.shape[0]))

    # sort S based on the first column of S (time order)
    sort_ind = np.lexsort(S[:,::-1].T)
    S1 = S[sort_ind[:m]]
    S2 = S[sort_ind[m:]]

    s1_mean = np.mean(S1[:, -1])
    s2_mean = np.mean(S2[:, -1])
    s1_var = np.var(S1[:, -1])
    s2_var = np.var(S2[:, -1])
    delta_prime = 0.025
    # 0.05 - quantile for normal distribution
    if S.shape[0] > 29:
        t_alpha = 2.048
    elif S.shape[0] > 14:
        t_alpha = 2.145
    else:
        t_alpha = 2.179
    Sw = (s1_var*(S1.shape[0]-1)+s2_var*(S2.shape[0]-1))/(S.shape[0]-2)
    t = (np.abs(s1_mean - s2_mean) - delta_prime) /(Sw * np.sqrt(1.0/S1.shape[0] + 1.0/S2.shape[0]))
    # print "narrow scale",t
    if t > t_alpha:
        result = True
        # print result
        t_star = S1[-1,0]
    else:
        result = False
        t_star = 0

    return result,t_star


def model_adaptation(batchdata,clf,t,t_star):
    dim = batchdata[0]["xtrain"].shape[1]
    batch_volume = batchdata[0]["xtrain"].shape[0]
    volume = (t-t_star)* batch_volume
    x = np.zeros((volume,dim))
    y = np.zeros(volume)
    for ind in range(t-t_star):
        x[(ind*batch_volume):((ind+1)*batch_volume),:] = batchdata[ind+t_star]["xtrain"]
        y[(ind * batch_volume):((ind + 1) * batch_volume)] = batchdata[ind+t_star]["ytrain"]
    clf.fit(x, y)
    return clf


def feature_generation(batchdata, windowsize, classifier, theoretical = False):

    # initializaton
    scaler = StandardScaler()
    clf, D, acc = initialization(batchdata[0],classifier,scaler)

    for t in range(1,len(batchdata)):

        # accumulate detection features to build S and T
        if classifier != 'neural networks':
            acc.append(clf.score(batchdata[t]["xtest"], batchdata[t]["ytest"]))
            D.extend(cross_val(clf, data=(batchdata[t]["xtest"], batchdata[t]["ytest"]), t=t + 1))
        else:
            acc.append(clf.score(scaler.transform(batchdata[t]["xtest"]), batchdata[t]["ytest"]))
            D.extend(cross_val(clf, data=(scaler.transform(batchdata[t]["xtest"]), batchdata[t]["ytest"]), t=t + 1))
        if len(D) < 2*windowsize:
            # acc.append(clf.score(batchdata[t]["xtest"], batchdata[t]["ytest"]))
            continue

        # drift detecion
        result,t_star = drift_detection(D,windowsize,t)

        if theoretical == False:
            if result:
                clf = model_adaptation(batchdata,clf,int(t),int(t_star))
                D = []
        else:
            if t == 50 or t == 100 or t == 150:
                print "pause"
                t_star = t+2
                clf = model_adaptation(batchdata, clf, int(t_star), int(t))

    return np.array(acc).reshape(-1), D


def drift_detection(D,windowsize,t):
    """
    detection frame-work on D, can be substitute with other detection algorithms
    :param D: detection feature list
    :param windowsize: appropriate windowsize to accumulate features and split window
    :param t: current time
    :return: outputs result (True/False) and the drift point t_star
    """
    T, T_minus, S = TS_split(D, windowsize)
    result = False
    t_star = t+1
    broadResult = DD_broad_scale(T, S)
    # print t
    if broadResult:
        print ("broad scale: %d " % (t + 1))
        result, t_star = DD_narrow_scale(T_minus)
        if result:
            print ("at time step %d. find narrow scale: %d " % (t+1, t_star + 1))
    return result, t_star




