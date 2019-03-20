# -*- coding: utf-8 -*-
"""
@Date: 2019-01-29 20:10:29
@author: CarySun
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



def corr_filter(dataset, r=1.0, headmap=False):
    """
    Feature selection. Through correlation matrix, find out the variables whose correlation exceeds r.
    Parameters
    ----------
    0712
    dataset : pandas.Dataset
    r : sequence of percentile values
        percentile or percentiles to find score at
    axis : float
        correlation coefficient
    Returns
    -------
    corr_feature: list
       	list of tuples, each tuple contains two related features
        pandas can transformate it to dataframe
 	"""
    corr_mat_bool = ~(dataset.corr() > r)

    if headmap:

        sn.heatmap(corr_filter)

    mask = np.triu(np.ones(corr_mat_bool.shape, dtype=bool))

    corr_mat_bool = corr_mat_bool + mask

    corr_feature = []
    for indexs in corr_mat_bool.index:
        for i in range(len(corr_mat_bool.loc[indexs].values)):
            if (corr_mat_bool.loc[indexs].values[i] == False):
                corr_feature.append((indexs, corr_mat_bool.index[i]))

    return corr_feature


if __name__=='__main__':
    