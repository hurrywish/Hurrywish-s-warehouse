import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from secure_SD2_01 import secure_SD2_01
from chi2_contingency_test import chi2_contingency_test
from RandomTree_fillna import regr_fillna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestRegressor

# Simple Process of the original data, mainly drop duplicates
data = pd.read_csv('/Users/hurrywish/Desktop/Jupyter/rankingcard.csv', index_col=0)
data.drop_duplicates(inplace=True)
data.reset_index(drop=True, inplace=True)

# Use RandomForest to fill nan values
data1 = data.copy()
data2 = regr_fillna(data1)
data2.drop(index=data2[data2.age == 0].index, inplace=True)
data2.drop(index=data2[data2['NumberOfTimes90DaysLate'] > 90].index, inplace=True)
data2.reset_index(drop=True, inplace=True)

# setting targets
label = data2['SeriousDlqin2yrs']
sample = data2[[i for i in data2.columns if i != 'SeriousDlqin2yrs']]

# up-sample
sm = SMOTE(random_state=0)
sample_sm, label_sm = sm.fit_sample(sample, label)
# print(sample_sm.shape, label_sm.shape)

# setting train set and test set
x_train, x_test, y_train, y_test = train_test_split(sample_sm, label_sm, train_size=0.8, random_state=0)
data_train = pd.concat([y_train, x_train], axis=1)
data_test = pd.concat([y_test, x_test], axis=1)
# print(data_test.isnull().sum())

# binning
columns_iv = dict()
columns = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',

]
for column in columns:
    # column = 'age'
    data_bin = data_train.copy()
    cats, retbins = pd.qcut(data_bin[column], retbins=True, q=20, duplicates='drop')
    cats_name = str(column + '_cats')
    data_bin[cats_name] = cats

    SD2_0 = data_bin[data_train['SeriousDlqin2yrs'] == 0].groupby([cats_name]).count()['SeriousDlqin2yrs']
    SD2_1 = data_bin[data_train['SeriousDlqin2yrs'] == 1].groupby(cats_name).count()['SeriousDlqin2yrs']

    df = pd.DataFrame({'low_limit': retbins[0:-1],
                       'high_limit': retbins[1:],
                       'SD2_0': SD2_0.values,
                       'SD2_1': SD2_1.values
                       })

    # make sure that each row has SD2_0 and SD2_1 value
    df = secure_SD2_01(df)
    # binning by chi2_contingency_test
    len_data, iv_list = chi2_contingency_test(df)
    iv_increase = [(iv_list[i + 1] - iv_list[i]) / iv_list[i] for i in range(len(iv_list) - 1)]
    # iv_increase_rate = [(iv_increase[i + 1] - iv_increase[i]) / iv_increase[i] for i in range(len(iv_increase) - 1)]

    global best_value
    for i, j in enumerate(iv_increase):
        if j < 10 ** -3:
            best_value = len_data[i]
            break
    # best_value = np.argsort(iv_increase_rate)[-1]+2

    bins = chi2_contingency_test(df, number=best_value, export_bins=True)
    columns_iv[column] = (len_data, iv_list, best_value, bins)

# print(columns_iv)

for i in columns_iv:
    plt.plot(columns_iv[i][0], columns_iv[i][1])
    plt.axvline(x=columns_iv[i][2], c='r', ls='--')
    plt.title(i)
    plt.xticks(columns_iv[i][0])
    plt.show()

auto_col_bins = dict()
for i in columns_iv:
    auto_col_bins[i] = columns_iv[i][2]

auto_bins = dict()
for i in columns_iv:
    auto_bins[i] = columns_iv[i][3]

manual_bins = {
    'NumberOfTime30-59DaysPastDueNotWorse': [0, 1, 2, 13],
    'NumberOfTimes90DaysLate': [0, 1, 2, 17],
    'NumberOfTime60-89DaysPastDueNotWorse': [0, 1, 2, 8],
    'NumberOfDependents': [0, 1, 2, 3],
    'NumberRealEstateLoansOrLines': [0, 5, 8, 11, 58]
}

manual_bins = {k: [-np.inf, *manual_bins[k], np.inf] for k in manual_bins}
auto_bins = {k: [-np.inf, *auto_bins[k], np.inf] for k in auto_bins}

bins = {**manual_bins, **auto_bins}


# print(bins)


def bins_woe(data, bins):
    woe_dict = dict()
    woe_dict1 = dict()
    for i in bins:
        data_woe = data.copy()
        column = str(i) + '_cats'
        data_woe[column] = pd.cut(data_woe[i], bins=bins[i])
        data_middle = data_woe.groupby(['SeriousDlqin2yrs'])[column].value_counts().unstack(0).fillna(0.1)
        data_middle['woe'] = np.log((data_middle[0] / data_middle[0].sum()) / (data_middle[1] / data_middle[1].sum()))
        woe_dict[i] = {i: k for i, k in zip(data_middle.index, data_middle['woe'])}

    data_map = data.copy()
    for i in data_map.columns[1:]:
        data_map[i] = data_map[i].map(woe_dict[i])
    return data_map


data_train_woe = bins_woe(data_train, bins).reset_index(drop=True)
data_test_woe = bins_woe(data_test, bins).reset_index(drop=True)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# print(data_test_woe.isnull().sum())
# breakpoint()

x_train_woe = data_train_woe.iloc[:, 1:]
y_train_woe = data_train_woe.iloc[:, 0]
x_test_woe = data_test_woe.iloc[:, 1:]
y_test_woe = data_test_woe.iloc[:, 0]
# print(x_test_woe.isnull().sum())


best_solver = 'saga'
best_penalty = 'l2'
best_C = 1.11
max_iter = 200

LR = LogisticRegression(penalty=best_penalty, solver=best_solver, C=best_C, max_iter=max_iter, random_state=0)

LR.fit(x_train_woe, y_train_woe)
y_pred = LR.predict(x_test_woe)
train_score = LR.score(x_train_woe, y_train_woe)
test_score = LR.score(x_test_woe, y_test_woe)
# test_score = cross_val_score(x_test_woe, y_test_woe,cv=5).mean()
accuracy = accuracy_score(y_test_woe, y_pred)
recall = recall_score(y_test_woe, y_pred)
f1 = f1_score(y_test_woe, y_pred)

print('train_score:', train_score,
      'test_score:', test_score,
      'accuracy:', accuracy,
      'recall', recall,
      'f1', f1
      )

import scikitplot as skplt

y_prob = pd.DataFrame(LR.predict_proba(x_test_woe))

skplt.metrics.plot_roc(y_test, y_prob, plot_micro=False, plot_macro=False)
plt.show()

# train_scores = list()
# c_range = np.arange(0.1, 3.1, 0.1)
# for c in c_range:
#     LR = LogisticRegression(solver=best_solver, C=c, random_state=0, n_jobs=-1)
#     LR.fit(x_train_woe, y_train_woe)
#     train_score = LR.score(x_train_woe, y_train_woe)
#     train_scores.append(train_score)
#
# plt.plot(c_range, train_scores)
# plt.xticks(c_range)
# plt.show()

from sklearn.metrics import roc_auc_score, roc_curve, auc


# y_prob=pd.DataFrame(LR.predict_proba(x_test))
# print(y_prob)
# fpr, tpr, threshold = roc_curve(y_test, y_pred)
# plt.plot(fpr,tpr)
# plt.show()


def score_reporting(data, bins, constants):
    woe_dict = dict()
    for i in bins:
        data_woe = data.copy()
        column = str(i) + '_cats'
        data_woe[column] = pd.cut(data_woe[i], bins=bins[i])
        data_middle = data_woe.groupby(['SeriousDlqin2yrs'])[column].value_counts().unstack(0).fillna(0.1)
        data_middle['woe'] = np.log((data_middle[0] / data_middle[0].sum()) / (data_middle[1] / data_middle[1].sum()))
        woe_dict[i] = {i: k for i, k in zip(data_middle.index, data_middle['woe'])}
    data_map = data.copy()
    for i in data_map.columns[1:]:
        data_map[i] = data_map[i].map(woe_dict[i]) * constants[i]
    return data_map


# Rating
# Score=A-B*log(odds)
intercept = LR.intercept_
B = -(600 - 620) / (np.log(1 / 60) - np.log(1 / 30))
A = 600 + B * np.log(1 / 60)

base_score = A - B * intercept

data3 = data2.copy()

cnsts = {columns: -B * LR.coef_[0][i] for i, columns in enumerate(x_train_woe.columns)}

data4 = score_reporting(data3, bins, cnsts)
data4['base_score'] = base_score[0]
data4['overall_score'] = 0
data4['middle_value'] = 0


for column in data_train_woe.columns:
    data4['middle_value'] += data4[column]

data4['overall_score'] = data4['base_score'] - data4['middle_value']
data4.drop(labels=['middle_value'],axis=1,inplace=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data4.to_csv('result.csv')
