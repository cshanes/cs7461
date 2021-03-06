from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from hw1_util import create_two_spirals, get_hill_valley_data


X, y = create_two_spirals()

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Set the parameters by cross-validation
# tuned_parameters = [{'n_estimators': range(20,200), 'learning_rate': np.arange(0.1, 2.5, 0.1)}]

classifiers = [
    # KNeighborsClassifier(3),
    SVC(C=1)]
    # DecisionTreeClassifier(max_depth=10),
    # MLPClassifier(hidden_layer_sizes=(100, 20))]

mlp_layer_sizes = []
for i in range(1, 200):
    for j in range(1, 50):
        mlp_layer_sizes.append((i, j))

tuned_parameters = [
    # {'n_neighbors': range(1,20)},
    {'gamma': np.arange(0, 5, 0.1), 'C': [1]},
    # {'max_depth': range(0, 100)},
    # {'hidden_layer_sizes': mlp_layer_sizes, 'alpha': np.arange(0, 1, 0.01), }
]

for i in range(len(classifiers)):
    print("# Tuning hyper-parameters for roc_auc")
    print()
    clf = classifiers[i]
    params = tuned_parameters[i]

    clf = GridSearchCV(clf, params, cv=10, scoring='roc_auc')
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    clf.fit(X_test, y_test)
    print(clf.best_params_)
    print()