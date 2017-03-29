import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def create_two_spirals():
    spiral_data = pd.read_csv('twoSpirals.csv')
    return spiral_data.drop('class', axis=1).values, spiral_data.loc[:, 'class'].values


def get_hill_valley_data():
    data = pd.read_csv('https://raw.githubusercontent.com/rhiever/Data-Analysis-and-Machine-Learning-Projects/master'
                       '/tpot-demo/Hill_Valley_without_noise.csv.gz', sep='\t', compression='gzip')
    X = data.drop('class', axis=1).values
    y = data.loc[:, 'class'].values
    return X, y


def get_all_classifiers():
    return [
        get_knn(),
        get_svc_linear(),
        get_svc_rbf(),
        get_dtree(),
        get_mlp(),
        get_adaboost()
    ]


def get_knn():
    return KNeighborsClassifier(n_neighbors=3)


def get_svc_linear():
    return SVC(kernel="linear", C=0.025)


def get_svc_rbf():
    return SVC(gamma=0.2, C=1)


def get_dtree():
    return DecisionTreeClassifier(max_depth=10)


def get_mlp():
    return MLPClassifier(hidden_layer_sizes=(100, 20))


def get_adaboost():
    return AdaBoostClassifier(n_estimators=125, learning_rate=0.3, random_state=0, base_estimator=LogisticRegression())


def get_all_untuned_classifiers():
    return [
        KNeighborsClassifier(),
        SVC(kernel='linear'),
        SVC(),
        DecisionTreeClassifier(),
        MLPClassifier(),
        AdaBoostClassifier()
    ]


def get_tuned_hill_valley_classifiers():
    return [
        KNeighborsClassifier(n_neighbors=1),
        SVC(kernel='linear', C=0.6),
        SVC(),
        DecisionTreeClassifier(max_depth=40),
        MLPClassifier(hidden_layer_sizes=(100, 20)),
        AdaBoostClassifier(base_estimator=LogisticRegression())
    ]


def measure_hill_valley_auc():
    # classifiers = get_all_untuned_classifiers()
    classifiers = get_tuned_hill_valley_classifiers()
    X, y = get_hill_valley_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
    for clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = roc_auc_score(y_test, y_pred)
        print(clf)
        print(score)

if __name__ == '__main__':
    measure_hill_valley_auc()
