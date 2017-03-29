import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import hw1_util
from sklearn.model_selection import train_test_split


classifiers = hw1_util.get_all_classifiers()
# X, y = hw1_util.create_two_spirals()
X, y = hw1_util.get_hill_valley_data()


for clf in classifiers:
    print(type(clf))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

    start_time = time.perf_counter()
    clf.fit(X_train, y_train)
    end_time = time.perf_counter()
    print('Fit time: ' + str(end_time - start_time))

    start_time = time.perf_counter()
    clf.predict(X_test)
    end_time = time.perf_counter()
    print('Predict time: ' + str(end_time - start_time))


