#!/usr/bin/env python
# coding=UTF-8
'''
@Author: CarySun
@Date: 2019-03-18 20:26:58
'''


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



def corr_filter(dataset, r=1.0, headmap=False):
    """Feature selection. Through correlation matrix, find out the variables whose correlation exceeds r.

    Parameters
    ----------
    dataset : pandas.DataSet
    r : sequence of percentile values
        percentile or percentiles to find score at
    axis : float
        correlation coefficient
    headmap : bool, optional

    Returns
    -------
    corr_feature: list
       	list of tuples, each tuple contains two related features
        pandas can transformate it to dataframe
            
    Attributes
    ----------
    variances_ : array, shape (n_features,)
        Variances of individual features.

    Examples
    --------
    The following dataset has integer features, two of which are the same
    in every sample. These are removed with the default setting for threshold::

        >>> X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
        >>> selector = VarianceThreshold()
        >>> selector.fit_transform(X)
        array([[2, 0],
               [1, 4],
               [1, 1]])
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
    