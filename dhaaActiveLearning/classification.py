from enum import Enum
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm import SVC
# import libopf_py # check https://github.com/dhaalves/LibOPF to use OPF with sklearn estimator API

from .dataset import Dataset


def SVM():
    return SVC(probability=True, gamma='auto', random_state=1, verbose=0)

def kNN():
    return KNeighborsClassifier(n_neighbors=4)

def RF():
    return RandomForestClassifier(n_estimators=10, random_state=1, verbose=0)

def NB():
    return GaussianNB()

class Classifier(Enum):
    SVM = {'name': 'SVM', 'instance': SVM}
    kNN = {'name': 'k-NN', 'instance': kNN}
    RF = {'name': 'RF',
          'instance': RF}
    NB = {'name': 'NB', 'instance': NB}

    @classmethod
    def get_default(cls):
        return Classifier.RF

    def get_name(self):
        return self.value['name']

    def get_instance(self):
        return self.value['instance']()

    @classmethod
    def get_names(cls):
        return list(map(lambda x: x.value.get('name'), cls))

    @classmethod
    def get_instances(cls):
        return list(map(lambda x: x.value.get('instance')(), cls))

    @classmethod
    def get_instance_from_name(cls, name):
        return next(filter(lambda x: x.value.get('name') == name, cls)).value.get('instance')()


def compute_accuracy_score(clf, x_test, y_test):
    return accuracy_score(y_test, clf.predict(x_test))


def compute_confusion_matrix(clf, x_test, y_test):
    return confusion_matrix(y_test, clf.predict(x_test))


def compute_cross_val_scores(clf, x, y, folds=5, scoring=None):
    scores = cross_val_score(clf, x, y, cv=folds, n_jobs=folds, scoring=scoring, verbose=0)
    # print(clf.__class__.__name__, scores, scores.mean(), scores.std())
    return scores


def compute_pca_scores(n_components, clf, dataset, scores_with_cv=False):
    assert isinstance(dataset, Dataset)
    pca = PCA(n_components=n_components)
    dataset.features = pca.fit_transform(dataset.features)
    if scores_with_cv:
        scores = np.mean(compute_cross_val_scores(clf, dataset.features, dataset.labels, folds=5))
    else:
        x_train, y_train, x_test, y_test, _, _ = dataset.get_split()
        clf.fit(x_train, y_train)
        scores = clf.score(x_test, y_test)
    Classifier.RF.get_instance().rese
    return scores


def grid_search_svm(x_train, y_train, x_test, y_test):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    scores = ['precision', 'recall']
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5)
    clf.fit(x_train, y_train)
    print(clf.best_params_)
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    print(compute_pca_scores(10, RandomForestClassifier(n_estimators=10), Dataset('LEA-53'), scores_with_cv=False))
    print(compute_pca_scores(10, RandomForestClassifier(n_estimators=10), Dataset('LEA-53'), scores_with_cv=True))
