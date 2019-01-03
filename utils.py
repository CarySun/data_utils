# coding: utf-8


def corr_filter(dataset, r=1, headmap=False):
    
    corr_mat_bool = ~(dataset.corr() > r)

    if headmap:
        import seaborn
        seaborn.heatmap(corr_filter)

    import numpy
    mask = numpy.triu(numpy.ones(corr_mat_bool.shape, dtype=bool))
    
    corr_mat_bool = corr_mat_bool + mask
    
    temp = []
    for indexs in corr_mat_bool.index:
        for i in range(len(corr_mat_bool.loc[indexs].values)):
            if (corr_mat_bool.loc[indexs].values[i] == False):
                temp.append((indexs, corr_mat_bool.index[i]))
    
    return temp 