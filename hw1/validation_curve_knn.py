import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.neighbors import KNeighborsClassifier
from hw1_util import create_two_spirals, get_hill_valley_data

X, y = get_hill_valley_data()

param_range = range(1, 25)
train_scores, test_scores = validation_curve(
    KNeighborsClassifier(), X, y, param_name='n_neighbors', param_range=param_range,
    cv=10, scoring='accuracy', n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title('Validation Curve with kNN')
plt.xlabel('# Neighbors')
plt.ylabel('Score')
plt.ylim(0.2, 1.1)
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
plt.show()
