import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.neural_network import MLPClassifier
from hw1_util import get_hill_valley_data

X, y = get_hill_valley_data()

hidden_layer_sizes = []
for i in range(1, 50):
    hidden_layer_sizes.append((50, i))

# train_scores, test_scores = validation_curve(
#     MLPClassifier(), X, y, param_name="hidden_layer_sizes", param_range=hidden_layer_sizes,
#     cv=5, scoring="accuracy", n_jobs=1)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# plt.title("Validation Curve with MLPClassifier")
# plt.xlabel("Hidden layers")
# plt.ylabel("Score")
# plt.ylim(0.0, 1.1)
# lw = 2
#
# param_range = range(1, 50)
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

# param_range = [(50, 5), (60, 5), (70, 5), (80, 5), (90, 5), (100, 5), (110, 5), (120, 5), (130, 5), (140, 5), (150, 5), (160, 5), (170, 5), (180, 5), (190, 5), (200, 5)]
# train_scores, test_scores = validation_curve(
#     MLPClassifier(), X, y, param_name="hidden_layer_sizes", param_range=param_range,
#     cv=5, scoring="accuracy", n_jobs=1)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# plt.title("Validation Curve with MLPClassifier")
# plt.xlabel("Hidden units (5 layers)")
# plt.ylabel("Score")
# plt.ylim(0.0, 1.1)
# lw = 2
#
# param_range = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
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


######## alpha #########
# param_range = np.arange(0, 1, 0.01)
# train_scores, test_scores = validation_curve(
#     MLPClassifier(hidden_layer_sizes=(100, 5)), X, y, param_name="alpha", param_range=param_range,
#     cv=5, scoring="accuracy", n_jobs=1)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# plt.title("Validation Curve with MLPClassifier")
# plt.xlabel("L2 penalty")
# plt.ylabel("Score")
# plt.ylim(0.0, 1.1)
# lw = 2
#
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


# ########## learning rate ##########
# param_range = ['constant', 'invscaling', 'adaptive']
# train_scores, test_scores = validation_curve(
#     MLPClassifier(hidden_layer_sizes=(100, 5), solver='sgd'), X, y, param_name="learning_rate", param_range=param_range,
#     cv=5, scoring="accuracy", n_jobs=1)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# plt.title("Validation Curve with MLPClassifier")
# plt.xlabel("L2 penalty")
# plt.ylabel("Score")
# plt.ylim(0.0, 1.1)
# lw = 2
#
# param_range = [1, 2, 3]
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
# print('plot time bro')
# plt.show()


# MOMENTUM
# param_range = np.arange(0, 1.1, 0.1)
# train_scores, test_scores = validation_curve(
#     MLPClassifier(hidden_layer_sizes=(100, 5), solver='sgd'), X, y, param_name="momentum", param_range=param_range,
#     cv=5, scoring="accuracy", n_jobs=1)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# plt.title("Validation Curve with MLPClassifier")
# plt.xlabel("Momentum")
# plt.ylabel("Score")
# plt.ylim(0.0, 1.1)
# lw = 2
#
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
# print('plot time bro')
# plt.show()


# LEARNING RATE INIT
param_range = np.arange(0.001, 0.05, 0.001)
train_scores, test_scores = validation_curve(
    MLPClassifier(hidden_layer_sizes=(100, 5), solver='sgd', learning_rate='constant'), X, y, param_name="learning_rate_init", param_range=param_range,
    cv=10, scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with MLPClassifier")
plt.xlabel("Learning Rate")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2

plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
print('plot time bro')
plt.show()
