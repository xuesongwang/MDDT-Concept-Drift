# -*- coding: utf-8 -*-
"""
 Multiscale drift detection algorithm

"""

import numpy as np
import pandas as pd
import warnings

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from numpy.random import randint
from scipy.stats import t as student

warnings.filterwarnings('ignore')

class DriftDetection:
    def __init__(self,data = None,classifier = 'KNN',parameter = None,initialsize = 0):
        self.DriftIndicators = None
        self.data = data
        self.initialsize = initialsize
        self.windowsize = parameter["windowsize"]
        self.batchsize = 10
        self.standardize = None
        if classifier == 'decision tree':
            clf = DecisionTreeClassifier()
        elif classifier == 'SVM':
            clf = SVC()
            self.standardize = StandardScaler()
        else:
            clf = KNeighborsClassifier()
        self.clf = clf

    def initialization(self,initialsize = 50,data = None):
        """
        initialize accuracy list, detection feature list, and a primary classifier
        :param firstbatch: first batch data (batchdata[0])
        :return: a classifier, detection feature D, and accuracy list acc
        """
        if data == None:
            data = self.data[:initialsize]
        if self.standardize != None:
            x = self.standardize.fit_transform(data[:,:-1])
        else:
            x = data[:,:-1]
        y = data[:,-1]
        clf = self.clf
        clf.fit(x, y)
        y_pred = clf.score(x,y)
        self.DriftIndicators = pd.Series(y_pred)
        self.clf = clf
        self.initialsize = initialsize

    def TS_split(self,D,windowsize):
        """
        split D into T,S . S is sampled from S_telta. when the result of broad_scale is true,
        check in detail the last window before T (T_minus)
        :param D:
        :param windowsize:
        :return: current test window T, last test window T_minus, sub-stationary window S
        """
        T = D[-windowsize:]
        S_telta =D[:-windowsize]

        # select index for S
        rand_ind = randint(0, S_telta.shape[0], windowsize)
        # rand_ind = weighted_sampling(S_telta.shape[0],windowsize)
        S = S_telta[S_telta.index.values[rand_ind]]
        return T, S

    def calBeta(self,alpha,n):
        TratioSquared = (student.ppf(alpha, n - 2) / student.ppf(alpha, 2 * n - 2)) ** 2
        beta = 1 + ((n - 4) * 1.0 / (n - 2)) * TratioSquared
        beta = 1.0 / beta
        # print "m = %d; n = %d, beta= %.4f" % (beta * n, n, beta)
        return beta

    def tTest(self,x,y,delta = 0.0,alpha = 0.95):
        x_mean = x.mean()
        x_sigma = x.var()
        n1 = x.shape[0]
        y_mean = y.mean()
        y_sigma = y.var()
        n2 = y.shape[0]
        sigmaSquared = ((n1-1)*x_sigma + (n2-1)*y_sigma)/(n1+n2-2)
        t = (np.abs(x_mean - y_mean)- delta)/np.sqrt(sigmaSquared*(1.0/n1 + 1.0/n2))
        t_sig = student.ppf(alpha,n1+n2-2)
        # print t,t_sig, t > t_sig
        return t > t_sig,t

    def DD_narrow_scale(self,S):
        """
        :param S:
        :return:
        """
        beta = 1-self.calBeta(0.95, S.shape[0])
        m = int(np.floor(beta * S.shape[0]))
        # sort S based on the first column of S (time order)
        S1 = S[:m]
        S2 = S[m:]
        result,_ = self.tTest(S1,S2)
        drift_time = S1.index.values[-1] if result else None
        return result,drift_time

    def model_adaptation(self,t,t_star):
        clf = self.clf
        data = self.data
        sub_data = data[t_star:t]
        if self.standardize != None:
            x = self.standardize.transform(sub_data[:,:-1])
        else:
            x = sub_data[:,:-1]
        y = sub_data[:,-1]
        try:
            clf.fit(x, y)
            print (clf.score(x, y))
        except:
            print (clf.score(x,y))
        self.clf = clf

    def drift_detection(self, D):
        """
        detection frame-work on D, can be substitute with other detection algorithms
        :param D: detection feature list
        :param windowsize: appropriate windowsize to accumulate features and split window
        :param t: current time
        :return: outputs result (True/False) and the drift point t_star
        """
        windowsize = self.windowsize
        if D.shape[0] < 1.3*windowsize:
            return False,None
        T, S = self.TS_split(D, windowsize)
        result = False
        t_star = None
        broadResult ,_ = self.tTest(S, T)
        if broadResult:
            result, t_star = self.DD_narrow_scale(T)
            if result:
                print ("at time step %d. find narrow scale: %d " % (D.index.values[-1], t_star))
        return result,t_star

    def train_and_detect(self):
        # initializaton
        detected_time = []
        data = self.data
        clf = self.clf
        batchsize = self.batchsize
        DriftIndicators = self.DriftIndicators
        predicted = []
        for t in range(self.initialsize,data.shape[0],batchsize):
            sub_data= data[t:t+batchsize]
            if self.standardize != None:
                x = self.standardize.transform(sub_data[:,:-1])
            else:
                x = sub_data[:,:-1]
            y = sub_data[:,-1]
            result = clf.score(x,y)
            DriftIndicators[t] = result
            predicted.append(result)
            result,t_star = self.drift_detection(DriftIndicators)
            if result:
                self.model_adaptation(t,t_star)
                detected_time.append(t_star)
                DriftIndicators = DriftIndicators[-self.windowsize:]
            if len(DriftIndicators) > 500:
                DriftIndicators = DriftIndicators[DriftIndicators.shape[0]/2:]
        print (np.array(detected_time)/batchsize)
        # pd.Series(np.array(detected_time)).to_csv('stagger.csv',index=False)
        # plot_index = np.arange(0,len(predicted),len(predicted)/1000)
        # pd.Series(np.array(predicted)[plot_index]).to_csv('Result/average_accuracy_without_adaptation/interchangingRBF.csv',index=False)
        # pd.Series(np.array(predicted)).to_csv(
        #     'Result/average_accuracy_with_adaptation/interchangingRBF_raw.csv', index=False)
        return np.mean(predicted)

    def constant_adaptation(self):
        data = self.data
        clf = self.clf
        batchsize = self.batchsize
        predicted = []
        for t in range(self.initialsize, data.shape[0], batchsize):
            try:
                result = clf.score(data[t:t + batchsize, :-1], data[t:t + batchsize, -1])
                predicted.append(result)
                # clf.fit(data[t:t + batchsize, :-1], data[t:t + batchsize, -1])
            except:
                continue
        return np.mean(predicted)






