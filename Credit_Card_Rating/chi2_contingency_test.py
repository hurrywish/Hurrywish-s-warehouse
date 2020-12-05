# binning by chi2_contingency
def chi2_contingency_test(data, number=2, export_bins=False):
    import numpy as np
    import pandas as pd
    from scipy.stats import chi2_contingency

    data_chi2 = data.copy()
    iv_list = list()
    len_data = list()

    while len(data_chi2) > number:

        data_chi2['total'] = data_chi2['SD2_0'] + data_chi2['SD2_1']
        data_chi2['percentage'] = data_chi2['total'] / data_chi2['total'].sum()
        data_chi2['good%'] = data_chi2.SD2_0 / data_chi2.SD2_0.sum()
        data_chi2['bad%'] = data_chi2.SD2_1 / data_chi2.SD2_1.sum()
        data_chi2['woe'] = np.log(data_chi2['good%'] / data_chi2['bad%'])
        iv = np.sum((data_chi2['good%'] - data_chi2['bad%']) * data_chi2['woe'])
        iv_list.append(iv)
        len_data.append(len(data_chi2))

        p_values = list()
        for i in range(len(data_chi2) - 1):
            SD2_i = data_chi2.loc[i, ['SD2_0', 'SD2_1']]
            SD2_i_1 = data_chi2.loc[i + 1, ['SD2_0', 'SD2_1']]
            p_value = chi2_contingency([SD2_i, SD2_i_1])[1]
            p_values.append(p_value)

        index = np.argsort(p_values)[-1]
        # print(index)
        data_chi2.loc[index + 1, ['SD2_0']] += data_chi2.loc[index, ['SD2_0']]
        data_chi2.loc[index + 1,['SD2_1']] += data_chi2.loc[index,['SD2_1']]
        data_chi2.loc[index + 1, ['low_limit']] = data_chi2.loc[index, ['low_limit']]
        data_chi2.drop(index=index, axis=0, inplace=True)
        data_chi2.reset_index(drop=True, inplace=True)

    if export_bins == True:
        bins = sorted(list(set(np.ravel(data_chi2[['low_limit', 'high_limit']]))))
        return bins
    else:
        return len_data[::-1], iv_list[::-1]
