# -*- coding: utf-8 -*-
"""
@Date: 2019-03-19 09:33:01
@author: CarySun
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import nan_vis




def df_summary(df, mode='all', labels=None):
    df = df.copy()

    if mode in  ('all', 'base'):  # if mode ==  'base' or mode == 'all':
        print('\n', '{:*^60}'.format('Data overview'), '\n')
        print('Sample: {0}\tFeature numbers: {1}'.format(df.shape[0], (df.shape[1] - 1)), '\n')
        print('\n', '{:-^60}'.format('Describe'))
        print(df.describe().T)
        print('\n', '{:-^60}'.format('Data type'))
        print(df.dtypes)

    
    if mode in ('all', 'labels'):
        print('\n', '{:*^60}'.format('Label overview'), '\n')
        label_size = len(np.unique(df[labels]))
        print('Labels\' size: {0}'.format(label_size), '\n')
        if label_size > 10 and df[labels].dtype != 'object':
            print('this is regression labels')
            

        if label_size < 10 and df[labels].dtype == 'object':
            print('{:-^60}'.format('Labels count'))
            print(df[labels].value_counts())



    if mode in ('all', 'nan'):
        na_cols = df.isnull().sum(axis=0)
        na_lines = df.isnull().any(axis=1)

        print('\n', '{:*^60}'.format('NaN overview'))             
        print('\n', 'Total number of NA lines is: {0}'.format(na_lines.sum()), '\n')        
        print('{:-^60}'.format('NaN Features'))
        print(na_cols[na_cols!=0].sort_values(ascending=False))

    return 0




if __name__=='__main__':
    df = pd.read_csv('finviz.csv')
    df_summary(df, 'all',labels='Country')

    

    