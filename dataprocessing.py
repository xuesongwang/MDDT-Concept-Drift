# -*- coding: utf-8 -*-
"""
 data processing
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from numpy.random import randint
import json
from scipy.io import arff


def load_data(datapath, labelpath):
    x = pd.read_csv(datapath, sep=',| ', header=None,engine='python').values
    y = pd.read_csv(labelpath, sep=' ', header=None).values
    out = np.concatenate([x,y],axis=1)
    return out


def data_augmentation(data,batchsize,duplicate_rate):
    xtrain, ytrain, xtest, ytest = data
    batch = xtrain.shape[0]/batchsize
    duplic_batchsize = batchsize * duplicate_rate
    # initialization
    xtrain_aug = np.zeros((xtrain.shape[0]*duplicate_rate, xtrain.shape[1]))
    ytrain_aug = np.zeros((ytrain.shape[0]*duplicate_rate))
    xtest_aug = np.zeros((xtest.shape[0]*duplicate_rate, xtest.shape[1]))
    ytest_aug = np.zeros((ytest.shape[0]*duplicate_rate))


    # replicate data within every batch

    rand_ind = []
    # generate index sequence for many times
    for rand in range(duplicate_rate):
        rand_ind.append(randint(0, batchsize, batchsize))

    for i in range(batch):
        temp_xtrain = xtrain[i * batchsize:(i + 1) * batchsize, :]
        temp_ytrain = ytrain[i * batchsize:(i + 1) * batchsize]
        temp_xtest = xtest[i * batchsize:(i + 1) * batchsize, :]
        temp_ytest = ytest[i * batchsize:(i + 1) * batchsize]

        for j in range(duplicate_rate):
            xtrain_aug[(i*duplic_batchsize + batchsize*j):(i*duplic_batchsize+batchsize*(j+1)),:]\
                = temp_xtrain[rand_ind[j],:]
            ytrain_aug[(i*duplic_batchsize + batchsize*j):(i*duplic_batchsize+batchsize*(j+1))]\
                = temp_ytrain[rand_ind[j]]
            xtest_aug[(i*duplic_batchsize + batchsize*j):(i*duplic_batchsize+batchsize*(j+1)),:]\
                = temp_xtest[rand_ind[j], :]
            ytest_aug[(i*duplic_batchsize + batchsize*j):(i*duplic_batchsize+batchsize*(j+1))] \
                = temp_ytest[rand_ind[j]]

    out = (xtrain_aug,ytrain_aug,xtest_aug,ytest_aug)
    return out


def batch_split(data,batchsize):
    x, y = data
    batch = x.shape[0]/batchsize
    batchdata = []
    for i in range(batch-1):
        index = range(i*batchsize,(i+1)*batchsize)
        batchtrain_x = x[index,:]
        batchtrain_y = y[index]
        batchtest_x = x[np.array(index).reshape(-1)+batchsize,:]
        batchtest_y = y[np.array(index).reshape(-1)+batchsize]
        batchdata.append({"xtrain":batchtrain_x,
                          "ytrain":batchtrain_y,
                          "xtest":batchtest_x,
                          "ytest":batchtest_y
        })
    return batchdata


def data_preprocessing(outpath,batchsize):
    xpath, ypath = outpath
    data = load_data(xpath, ypath)
    duplicate_rate = 5
    # data = data_augmentation(data, batchsize=batchsize, duplicate_rate=duplicate_rate)
    # batchdata = batch_split(data, batchsize=batchsize)
    return data


def dataset_config(name):
    config_path = open('Config/'+name)
    jsonData = json.load(config_path)

    filepath = 'datasource/' + jsonData['filepath']
    xpath = filepath + name + '.data'
    ypath = filepath + name + '.labels'

    # special cases for irregular formats
    if name == 'sine1':
        xpath = filepath + name+'.csv'
        ypath = filepath + name +'.labels'
    if name == 'weather':
        xpath = filepath + 'NEweather_data.csv'
        ypath = filepath + 'NEweather_class.csv'
    if name == 'sea':
        xpath = filepath + 'SEA_training_data.csv'
        ypath = filepath +'SEA_training_class.csv'
    if name == 'covType':
        xpath = filepath + 'covType_data.csv'
        ypath = filepath + 'covType_label.csv'
    if name == 'Elec2':
        xpath = filepath + 'elec2_data.dat'
        ypath = filepath + 'elec2_label.dat'
    parameters = {
        "outpath":(xpath, ypath),
        "batchsize": jsonData['batchsize'],
        "windowsize": jsonData['windowsize'],
        "classifier":jsonData['classifier']
    }

    return parameters

