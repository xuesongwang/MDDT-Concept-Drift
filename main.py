# -*- coding: utf-8 -*-
"""
 Multi-scale Drift Detection Test(MDDT)
 author: xuesong wang
 2018-11-5
"""

from Algorithm.Multiscale_Drift_Detection_Tests import DriftDetection
import dataprocessing as dp
import numpy as np
import time


if __name__ == '__main__':

    # assign parameters for every data set
    # parameters = dp.dataset_config('sea')
    # parameters = dp.dataset_config('sine1')
    # parameters = dp.dataset_config('sine2')
    parameters = dp.dataset_config('stagger')
    # parameters = dp.dataset_config('interchangingRBF')
    # parameters = dp.dataset_config('rotatingHyperplane')
    # parameters = dp.dataset_config('movingRBF')
    # parameters = dp.dataset_config('movingSquares')
    # parameters = dp.dataset_config('weather')
    # parameters = dp.dataset_config('elec2')
    # parameters = dp.dataset_config('coverType')
    # parameters = dp.dataset_config('poker')


    # data preprocessing including data reading, batch splitting and probable data augmenting
    data = dp.data_preprocessing(parameters["outpath"],parameters["batchsize"])

    print "total time step: %d" % (data.shape[0]/parameters["batchsize"])

    start = time.clock()
    MDDT = DriftDetection(data = data, parameter=parameters)
    MDDT.initialization()
    acc = MDDT.train_and_detect()
    # acc = MDDT.constant_adaptation()
    end = time.clock()
    print "time:%f"%(end-start)
    print "average accuracy:  %s %%" % (100 * acc)

    # plot features and accuracy
    # plot_features(acc,D)



