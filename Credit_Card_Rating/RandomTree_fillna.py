import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

# dataset = load_boston()
# x_full = dataset.data
# y_full = dataset.target
# rows = x_full.shape[0]
# columns = x_full.shape[1]
#
# size = columns * rows
# missing_rate = 0.2
# n_missing = int(np.floor(size * missing_rate))
# rs = np.random.RandomState(1)
#
# x_processed = np.ravel(x_full)
# missing_number = rs.randint(0, size + 1, n_missing)
# x_processed[missing_number] = np.nan
# x_processed = x_processed.reshape(rows, columns)
#
# data_x = pd.DataFrame(x_processed)


def regr_fillna(data):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    data = pd.DataFrame(data).reset_index(drop=True).copy()

    nan_values = list()
    for i in data.columns:
        nan_value = data[i].isnull().sum()
        nan_values.append(nan_value)
    nan_values = list(np.argsort(nan_values))
    print(nan_values)

    while nan_values:
        columns_fill_value = data.columns[nan_values.pop(0)]
        columns_fill_0 = [i for i in data.columns if i != columns_fill_value]

        if data[columns_fill_value].isnull().sum() == 0:
            print(columns_fill_value, '无空值')
        else:
            new_data = data.copy()
            new_data[columns_fill_0] = new_data[columns_fill_0].fillna(0)
            sample = new_data[columns_fill_0]
            label = new_data[columns_fill_value]

            y_train = label[label.notnull()]
            y_test = label[label.isnull()]
            x_train = sample.iloc[y_train.index]
            x_test = sample.iloc[y_test.index]

            regr = RandomForestRegressor(n_estimators=100,n_jobs=-1)
            regr.fit(x_train, y_train)
            y_pred = regr.predict(x_test)
            data.loc[y_test.index,[columns_fill_value]] = y_pred
            train_score = regr.score(x_train, y_train)
            print('%s列，得分:%0.2f' % (columns_fill_value, train_score))

    return data








# data1 = regr_fillna(x_processed)
# print(data1)
