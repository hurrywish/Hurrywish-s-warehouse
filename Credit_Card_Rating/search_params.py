

# def Random_Search_param_opt(x, y):
#     params = {'C': np.arange(0.01, 5.01, 0.01),
#               'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
#               'max_iter': np.arange(100, 10000, 100),
#               'penalty':['l2']
#               }
#     LR = LogisticRegression(random_state=0)
#     clf = RandomizedSearchCV(LR, param_distributions=params, scoring='recall', cv=5, n_iter=10, n_jobs=-1,
#                              random_state=0)
#     clf.fit(x, y)
#     return clf.best_params_
#
# best_params = Random_Search_param_opt(x_train_woe, y_train_woe)
# print(best_params)

# def Random_Search_param_opt(x, y):
#     params = {'C': np.arange(0.01, 5.01, 0.01),
#               'solver': ['saga','liblinear'],
#               'max_iter': np.arange(100, 10000, 100),
#               'penalty':['l1']
#               }
#     LR = LogisticRegression(random_state=0)
#     clf = RandomizedSearchCV(LR, param_distributions=params, scoring='recall', cv=5, n_iter=10, n_jobs=-1,
#                              random_state=0)
#     clf.fit(x, y)
#     return clf.best_params_

# best_params = Random_Search_param_opt(x_train_woe, y_train_woe)
# print(best_params)