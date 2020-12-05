# make sure that all the bins contain SD2_0 and SD2_1 value (not 0)
def secure_SD2_01(data):
    data_0_1 = data.copy()
    for i in range(len(data_0_1) - 1, 0, -1):
        SD2_0_value = data_0_1['SD2_0'].iloc[i]
        SD2_1_value = data_0_1['SD2_1'].iloc[i]
        if SD2_0_value == 0 or SD2_1_value == 0:
            data_0_1.loc[i - 1, ['SD2_0']] += SD2_0_value
            data_0_1.loc[i - 1, ['SD2_1']] += SD2_1_value
            data_0_1.loc[i - 1, ['high_limit']] = data_0_1.loc[i, ['high_limit']]
            data_0_1.drop(index=i, inplace=True)
            print('secure_SD2_01 deleted %s row' % i)

    if data_0_1['SD2_0'].iloc[0] == 0 or data_0_1['SD2_1'].iloc[0] == 0:
        data_0_1.loc[1, ['SD2_0']] += data_0_1.loc[0, ['SD2_0']]
        data_0_1.loc[1, ['SD2_1']] += data_0_1.loc[0, ['SD2_1']]
        data_0_1.loc[1, ['low_limit']] = data_0_1.loc[0, ['low_limit']]
        data_0_1.drop(index=0, inplace=True)
        print('secure_SD2_01 deleted 0 row')

    data_0_1.reset_index(drop=True, inplace=True)
    return data_0_1
