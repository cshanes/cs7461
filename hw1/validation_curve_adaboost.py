import matplotlib.pyplot as plt
from sklearn.calibration import *
from sklearn.ensemble import *
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import validation_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import *
from hw1_util import create_two_spirals, get_hill_valley_data

X, y = get_hill_valley_data()

# num estimators
# param_range = range(20,200)
# train_scores, test_scores = validation_curve(
#     AdaBoostClassifier(), X, y, param_name="n_estimators", param_range=param_range,
#     cv=10, scoring="accuracy", n_jobs=1)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# plt.title("Validation Curve with AdaBoostClassifier")
# plt.xlabel("Number of estimators")
# plt.ylabel("Score")
# plt.ylim(0.8, 1.1)
# lw = 2
# plt.plot(param_range, train_scores_mean, label="Training score",
#              color="darkorange", lw=lw)
# plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.2,
#                  color="darkorange", lw=lw)
# plt.plot(param_range, test_scores_mean, label="Cross-validation score",
#              color="navy", lw=lw)
# plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.2,
#                  color="navy", lw=lw)
# plt.legend(loc="best")
# plt.show()


# learning rate
# param_range = np.arange(0.1, 2.5, 0.1)
# train_scores, test_scores = validation_curve(
#     AdaBoostClassifier(), X, y, param_name="learning_rate", param_range=param_range,
#     cv=10, scoring="accuracy", n_jobs=1)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# plt.title("Validation Curve with AdaBoostClassifier")
# plt.xlabel("Learning rate (150 estimators)")
# plt.ylabel("Score")
# plt.ylim(0.7, 1.1)
# lw = 2
# plt.plot(param_range, train_scores_mean, label="Training score",
#              color="darkorange", lw=lw)
# plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.2,
#                  color="darkorange", lw=lw)
# plt.plot(param_range, test_scores_mean, label="Cross-validation score",
#              color="navy", lw=lw)
# plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.2,
#                  color="navy", lw=lw)
# plt.legend(loc="best")
# plt.show()


param_range = [BaggingClassifier(), CalibratedClassifierCV(), DecisionTreeClassifier(),
               ExtraTreeClassifier(), GaussianNB(), GradientBoostingClassifier(),
               LogisticRegression(), LogisticRegressionCV(), RandomForestClassifier()]

train_scores, test_scores = validation_curve(
    AdaBoostClassifier(), X, y, param_name="base_estimator", param_range=param_range,
    cv=10, scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with AdaBoostClassifier")
plt.xlabel("Base estimator")
plt.ylabel("Score")
plt.ylim(0.2, 1.1)
lw = 2

labels = ['Bagging', 'Calibrated', 'DecisionTree',
          'ExtraTree', 'GaussianNB', 'GradBoost',
          'LogReg', 'LogRegCV', 'RandomForest']
param_range = range(0, len(param_range))
plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
plt.xticks(param_range, labels, rotation=50)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)

plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")

print('plot time bro')
plt.show()
