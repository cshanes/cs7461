import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import validation_curve, train_test_split
from sklearn.svm import SVC
from hw1_util import create_two_spirals, get_hill_valley_data

X, y = get_hill_valley_data()

# param_range = np.logspace(-6, 1, 5)
# train_scores, test_scores = validation_curve(
#     SVC(), X, y, param_name='gamma', param_range=param_range,
#     cv=10, scoring='accuracy', n_jobs=1)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# plt.title('Validation Curve with SVM')
# plt.xlabel('$\gamma$')
# plt.ylabel('Score')
# plt.ylim(0.0, 1.1)
# lw = 2
# plt.semilogx(param_range, train_scores_mean, label='Training score',
#              color='darkorange', lw=lw)
# plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.2,
#                  color='darkorange', lw=lw)
# plt.semilogx(param_range, test_scores_mean, label='Cross-validation score',
#              color='navy', lw=lw)
# plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.2,
#                  color='navy', lw=lw)
# plt.legend(loc='best')

param_range = np.arange(0.1, 3.0, 0.1)
train_scores, test_scores = validation_curve(
    SVC(kernel='linear'), X, y, param_name='C', param_range=param_range,
    cv=10, scoring='accuracy', n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title('Validation Curve with SVM')
plt.xlabel('C')
plt.ylabel('Score')
plt.ylim(0.98, 1.015)
lw = 2
plt.plot(param_range, train_scores_mean, label='Training score',
         color='darkorange', lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color='darkorange', lw=lw)
plt.plot(param_range, test_scores_mean, label='Cross-validation score',
         color='navy', lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color='navy', lw=lw)
plt.legend(loc='best')

print('plot time bro')
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
clf = SVC(kernel='linear', C=0.6)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = roc_auc_score(y_test, y_pred)
print(score)