# -*- coding: utf-8 -*-
"""
@Date: 2019-03-19 09:33:01
@author: CarySun
"""
import pandas as pd
import numpy as np


def nullity_sort(df, sort=None):
    """
    Sorts a DataFrame according to its nullity, in either ascending or descending order.

    :param df: The DataFrame object being sorted.
    :param sort: The sorting method: either "ascending", "descending", or None (default).
    :return: The nullity-sorted DataFrame.
    """
    if sort == 'ascending':
        return df.iloc[np.argsort(df.count(axis='columns').values), :]
    elif sort == 'descending':
        return df.iloc[np.flipud(np.argsort(df.count(axis='columns').values)), :]
    else:
        return df


def nullity_filter(df, filter=None, p=0, n=0):
    """
    Filters a DataFrame according to its nullity, using some combination of 'top' and 'bottom' numerical and
    percentage values. Percentages and numerical thresholds can be specified simultaneously: for example,
    to get a DataFrame with columns of at least 75% completeness but with no more than 5 columns, use
    `nullity_filter(df, filter='top', p=.75, n=5)`.

    :param df: The DataFrame whose columns are being filtered.
    :param filter: The orientation of the filter being applied to the DataFrame. One of, "top", "bottom",
    or None (default). The filter will simply return the DataFrame if you leave the filter argument unspecified or
    as None.
    :param p: A completeness ratio cut-off. If non-zero the filter will limit the DataFrame to columns with at least p
    completeness. Input should be in the range [0, 1].
    :param n: A numerical cut-off. If non-zero no more than this number of columns will be returned.
    :return: The nullity-filtered `DataFrame`.
    """
    if filter == 'top':
        if p:
            df = df.iloc[:, [c >= p for c in df.count(axis='rows').values / len(df)]]
        if n:
            df = df.iloc[:, np.sort(np.argsort(df.count(axis='rows').values)[-n:])]
    elif filter == 'bottom':
        if p:
            df = df.iloc[:, [c <= p for c in df.count(axis='rows').values / len(df)]]
        if n:
            df = df.iloc[:, np.sort(np.argsort(df.count(axis='rows').values)[:n])]
    return df

def df_summary(df, mode='all', show_num=5):
    df = df.copy()

    if mode in  ('all', 'base'):  # if mode ==  'base' or mode == 'all':
        print('{:*^60}'.format('Data overview'), '\n')
        print('Sample: {0}\tFeature numbers: {1}'.format(df.shape[0], (df.shape[1] - 1)), '\n')
        print('{:-^60}'.format('Glancing at Sample'))
        print(df.head(show_num))
        print('\n', '{:-^60}'.format('Describe'))
        print(df.describe().T)
        print('\n', '{:-^60}'.format('Data type'))
        print(df.dtypes)
        print('-' * 60)

    if mode in ('all', 'nan'):
        na_cols = df.isnull().sum(axis=0)
        print('{:*^60}'.format('NaN overview'), '\n')        
        print('NaN Feature:')
        print(na_cols[na_cols!=0].sort_values(ascending=False))
        print('-' * 30)
        print('valid records for each Cols:')
        print(df.count())
        print('-' * 30)    
        na_lines = df.isnull().any(axis=1)
        print('Total number of NA lines is: {0}'.format(na_lines.sum()))
        print('-' * 30)

    return 0


if __name__=='__main__':
    df = pd.read_csv('finviz.csv')
    df_summary(df, 'nan')
    