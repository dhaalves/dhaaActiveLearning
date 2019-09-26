import json
import warnings
from enum import Enum
from timeit import default_timer as timer

import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from tqdm import trange

warnings.simplefilter(action='ignore', category=FutureWarning)

from .classification import Classifier
from .dataset import Dataset


def unique_without_sorting(array):
    indexes = np.unique(array, return_index=True)[1]
    return [array[idx] for idx in sorted(indexes)]


def get_knn(x, neighbors=None, n_neighbors=2, return_graph=False):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')  # , metric='euclidean')
    nbrs.fit(x)
    knn = neighbors if neighbors is not None else x
    if not return_graph:
        knn_dist, knn_idx = nbrs.kneighbors(knn)
        return knn_dist, knn_idx
    else:
        return nbrs.kneighbors_graph(knn, mode='distance')


def get_mst_idx(x):
    knn_graph = get_knn(x, n_neighbors=len(x), return_graph=True)
    mst_array = minimum_spanning_tree(knn_graph).toarray()
    nonzero_indices = np.asarray(mst_array.nonzero())
    data_argsort = mst_array[mst_array.nonzero()].argsort()
    idx = nonzero_indices[:, data_argsort].flatten('F')
    return idx[::-1]


def cluster_data(x, n_clusters):
    # noise_samples = len(list(filter(lambda x: x == -1, labels)))
    # clustered_samples = list(filter(lambda x: x != -1, labels))
    kmeans = KMeans(n_clusters=int(n_clusters), random_state=1).fit(x)
    cluster_ids = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    return cluster_ids, cluster_centers


def get_clusters_dict(cluster_ids):
    clusters_dict = dict()
    y_unique = np.unique(cluster_ids)
    for c in y_unique:
        clusters_dict[c] = np.argwhere(cluster_ids == c).flatten()
        # np.random.shuffle(clusters_dict[c])
    return clusters_dict


def get_mst_cluster_dict(x, cluster_ids, as_edges=True):
    cluster_dict = get_clusters_dict(cluster_ids)
    for c in cluster_dict:
        cluster_samples = x[cluster_dict[c]]
        mst_idx = get_mst_idx(cluster_samples)
        if not as_edges:
            mst_idx = unique_without_sorting(mst_idx)
        cluster_dict[c] = cluster_dict[c][mst_idx]
    return cluster_dict


def get_rds_cluster_dict(x, cluster_ids, cluster_centers):
    cluster_dict = get_clusters_dict(cluster_ids)
    rds_cluster_dict = dict()
    for c in cluster_dict:
        cluster_samples = cluster_dict[c]
        if len(cluster_samples) <= 1:
            continue
        _, knn_idx = get_knn(x[cluster_samples], neighbors=cluster_centers[c:c + 1],
                             n_neighbors=len(cluster_samples))
        rds_cluster_dict[c] = cluster_samples[knn_idx].flatten()
    return rds_cluster_dict


def get_mst_cluster_idx(x, cluster_ids, as_edges=True):
    cluster_dict = get_mst_cluster_dict(x, cluster_ids, as_edges=as_edges)
    num_samples = sum(map(len, cluster_dict.values()))
    idx = np.empty(0, int)
    while idx.size < num_samples:
        for c in cluster_dict:
            cluster_samples = cluster_dict[c]
            if cluster_samples.size != 0:
                idx = np.append(idx, cluster_dict[c][0])
                cluster_dict[c] = np.delete(cluster_dict[c], 0)
    return idx


def get_mst_rds_cluster_idx(x, cluster_ids, cluster_centers):
    cluster_dict = get_mst_cluster_dict(x, cluster_ids, as_edges=False)
    root_idx = get_root_idx(x, cluster_centers)
    num_samples = sum(map(len, cluster_dict.values()))
    idx = np.empty(0, int)
    while idx.size < num_samples:
        for c in cluster_dict:
            cluster_samples = cluster_dict[c]
            if cluster_samples.size != 0:
                idx = np.append(idx, root_idx[c])
                idx = np.append(idx, cluster_dict[c][0])
                cluster_dict[c] = np.delete(cluster_dict[c], 0)
    return idx


def get_root_idx(x, cluster_centers):
    _, knn_idx = get_knn(x, neighbors=cluster_centers, n_neighbors=1)
    idx = knn_idx.flatten()
    return idx


def get_boundary_idx(x, cluster_ids, as_edges=True, order=None):
    bedges_dist, bedges_idx = get_bondary_edges(x, cluster_ids)
    if order != None:
        stack = np.column_stack((bedges_dist, bedges_idx))
        if order == 'desc':
            stack = stack[stack[:, 0].argsort()[::-1]]
        elif order == 'asc':
            stack = stack[stack[:, 0].argsort()]
        bedges_idx = stack[:, 1:].astype(int)
    idx = bedges_idx.flatten()
    if not as_edges:
        idx = unique_without_sorting(idx)
    return idx


def get_bondary_edges(x, cluster_ids):
    knn_dist, knn_idx = get_knn(x)
    bedges_idx = np.empty((0, 2), int)
    bedges_dist = np.empty((0, 1), float)
    for i, knn_pair in enumerate(knn_idx):
        if cluster_ids[knn_pair[0]] != cluster_ids[knn_pair[1]] and \
                not np.any(np.all(bedges_idx == np.flip(knn_pair, axis=0), axis=1)):
            bedges_idx = np.vstack((bedges_idx, knn_pair))
            bedges_dist = np.append(bedges_dist, knn_dist[i][1])
    return bedges_dist, bedges_idx


def get_root_samples(x, y, cluster_centers):
    idx = get_root_idx(x, cluster_centers)
    return x[idx], y[idx]


def get_boundary_samples(x, y, cluster_ids, as_edges=True, order='desc'):
    idx = get_boundary_idx(x, cluster_ids, as_edges=as_edges, order=order)
    return x[idx], y[idx]


def get_mst_cluster_samples(x, y, cluster_ids, as_edges=False):
    idx = get_mst_cluster_idx(x, cluster_ids, as_edges=as_edges)
    return x[idx], y[idx]


def get_samples(x, y, n_clusters=None, strategy=None):
    n_clusters = len(np.unique(y)) if n_clusters is None else n_clusters
    cluster_ids, cluster_centers = cluster_data(x, n_clusters)
    root_idx, x_initial, y_initial = get_initial_data(cluster_centers, x, y)
    x_pool = np.delete(x, root_idx, axis=0)
    y_pool = np.delete(y, root_idx, axis=0)
    cluster_ids = np.delete(cluster_ids, root_idx, axis=0)
    organized_data = np.empty(0, int)
    if strategy == AL_Strategy.MST_CLU_DDE:
        organized_data = get_mst_cluster_idx(x_pool, cluster_ids, as_edges=True)
    elif strategy == AL_Strategy.MST_CLU_RDS:
        organized_data = get_mst_cluster_dict(x_pool, cluster_ids, as_edges=False)
    elif strategy == AL_Strategy.MST_CLU_RDS2:
        organized_data = get_mst_rds_cluster_idx(x_pool, cluster_ids, cluster_centers)
    elif strategy == AL_Strategy.MST_CLU_DS:
        organized_data = get_mst_cluster_idx(x_pool, cluster_ids, as_edges=False)
    elif strategy == AL_Strategy.RDS:
        organized_data = get_rds_cluster_dict(x_pool, cluster_ids, cluster_centers)
    elif strategy == AL_Strategy.RBE:
        organized_data = get_boundary_idx(x_pool, cluster_ids, as_edges=True)
    elif strategy == AL_Strategy.DBE:
        organized_data = get_boundary_idx(x_pool, cluster_ids, as_edges=True, order='desc')
    elif strategy == AL_Strategy.MST_BE:
        organized_data = get_boundary_idx(x_pool, cluster_ids, as_edges=False)
        x_pool, y_pool = x_pool[organized_data], y_pool[organized_data]
        organized_data = get_mst_idx(x_pool)
    elif strategy in [AL_Strategy.EN, AL_Strategy.MS, AL_Strategy.LC]:
        while True:
            idx = np.random.choice(len(y), n_clusters, replace=False)
            x_initial, y_initial = x[idx], y[idx]
            if len(np.unique(y_initial)) > 1: break
        x_pool, y_pool = np.delete(x, idx, axis=0), np.delete(y, idx, axis=0)
    return organized_data, root_idx, x_initial, y_initial, x_pool, y_pool


def get_initial_data(cluster_centers, x, y):
    root_idx = get_root_idx(x, cluster_centers)
    n_clusters = len(cluster_centers)
    while len(np.unique(y[root_idx])) < 2:
        n_clusters += 1
        _, cluster_centers = cluster_data(x, n_clusters)
        root_idx = get_root_idx(x, cluster_centers)
    return root_idx, x[root_idx], y[root_idx]


def root_distance_based_selection_strategy(classifier, x, n_instances=1, **kwargs):
    y_root = kwargs.get("y_root")
    dic = kwargs.get("idx")
    query_idx = np.empty(0, int)

    n_samples_left = sum(map(len, dic.values()))
    while (query_idx.size < n_samples_left):
        for l in dic:
            idx = dic[l]
            if idx.size == 0: continue
            pred = classifier.predict(x[idx])
            sel = idx[np.where(pred != y_root[l])]
            if sel.size != 0:
                query_idx = np.append(query_idx, sel[0])
                dic[l] = idx[np.where(idx != sel[0])]
            else:
                query_idx = np.append(query_idx, idx[-1])
                dic[l] = idx[:-1]
            if query_idx.size == n_instances or query_idx.size == n_samples_left:
                return query_idx, dic

    return query_idx, dic


def sequencial_query_strategy(classifier, x, n_instances=1, **kwargs):
    query_idx = np.arange(n_instances)
    return query_idx, x[query_idx]


def sequencial_idx_query_strategy(classifier, x, n_instances=1, **kwargs):
    idx = kwargs.get("idx")
    idx = np.asarray(idx)
    return idx[:n_instances], idx[n_instances:]


def random_query_strategy(classifier, x, n_instances=1, **kwargs):
    query_idx = np.random.choice(range(len(x)), n_instances)
    return query_idx, x[query_idx]


def disagree_labels_edges_idx_query_strategy(classifier, x, n_instances=1, step=2, **kwargs):
    labeled_idx = kwargs.get("labeled_idx")
    idx = kwargs.get("idx")
    query_idx = np.empty(0, int)
    disagree_edges_idx = np.empty(0, int)
    r = int(len(idx) / step)
    for i in range(r):
        begin = i * step
        end = begin + step
        edge_idx = idx[begin:end]
        pred = classifier.predict(x[edge_idx])
        if np.all(np.in1d(pred[1:], pred[0], invert=True)):
            disagree_edges_idx = np.append(disagree_edges_idx, np.arange(begin, end))
            query_idx = np.append(query_idx,
                                  np.array([e for e in edge_idx if e not in labeled_idx and e not in query_idx]).astype(
                                      int))
        if query_idx.size >= n_instances:
            return query_idx[:n_instances], np.delete(idx, disagree_edges_idx)

    if query_idx.size < n_instances:
        #         print('Not enough edges with distinct labels, getting samples with biggest edges')
        edges_left = unique_without_sorting(np.delete(idx, disagree_edges_idx)[::-1])
        for e in edges_left:
            if e not in labeled_idx and e not in query_idx:
                query_idx = np.append(query_idx, e)
            if query_idx.size >= n_instances:
                return query_idx[:n_instances], np.delete(idx, disagree_edges_idx)

    return query_idx, np.empty(0, int)


class AL_Parameters():

    def __init__(self, dataset_name, classifier_name, strategy_name,
                 max_iterations=25, n_clusters=None, n_instances=None, interactive_labeling=False):
        self.dataset_name = dataset_name
        self.dataset = Dataset(dataset_name)
        self.classifier_name = classifier_name
        self.classifier = Classifier.get_instance_from_name(classifier_name)
        self.n_clusters = len(self.dataset.classes) * 2 if n_clusters is None else int(n_clusters)
        self.n_instances = len(self.dataset.classes) * 2 if n_instances is None else int(n_instances)
        self.max_iterations = int(max_iterations)
        self.strategy_name = strategy_name
        self.strategy = AL_Strategy.from_name(strategy_name)
        self.interactive_labeling = interactive_labeling

    def __repr__(self):
        return f'Dataset: {self.dataset.name} - Strategy: {self.strategy_name} - Classifier: {self.classifier_name}'

    @classmethod
    def fromdict(cls, dict):
        return AL_Parameters(dict['dataset_name'],
                             dict['classifier_name'],
                             dict['strategy_name'], 25,
                             dict['n_clusters'],
                             dict['n_instances'],
                             dict.get('interactive_labeling', False))
        # al_params = cls()
        # al_params.__dict__ = dict
        # return al_params

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


class AL_Strategy(Enum):
    EN = {'name': 'EN', 'query_strategy': entropy_sampling, 'classic': True}
    MS = {'name': 'MS', 'query_strategy': margin_sampling, 'classic': True}
    LC = {'name': 'LC', 'query_strategy': uncertainty_sampling, 'classic': True}
    EN_CLU = {'name': 'EN-CLU', 'query_strategy': entropy_sampling, 'classic': True}
    MS_CLU = {'name': 'MS-CLU', 'query_strategy': margin_sampling, 'classic': True}
    LC_CLU = {'name': 'LC-CLU', 'query_strategy': uncertainty_sampling, 'classic': True}
    RBE = {'name': 'RBE', 'query_strategy': disagree_labels_edges_idx_query_strategy, 'classic': False}
    DBE = {'name': 'DBE', 'query_strategy': disagree_labels_edges_idx_query_strategy, 'classic': False}
    MST_BE = {'name': 'MST-BE', 'query_strategy': disagree_labels_edges_idx_query_strategy, 'classic': False}
    MST_CLU_DS = {'name': 'MST-CLU-DS', 'query_strategy': sequencial_idx_query_strategy, 'classic': False}
    MST_CLU_DDE = {'name': 'MST-CLU-DDE', 'query_strategy': disagree_labels_edges_idx_query_strategy, 'classic': False}
    RDS = {'name': 'RDS', 'query_strategy': root_distance_based_selection_strategy, 'classic': False}
    MST_CLU_RDS = {'name': 'MST-CLU-RDS', 'query_strategy': root_distance_based_selection_strategy, 'classic': False}
    MST_CLU_RDS2 = {'name': 'MST-CLU-RDS2', 'query_strategy': disagree_labels_edges_idx_query_strategy,
                    'classic': False}

    def get_name(self):
        return self.value['name']

    def get_query_strategy(self):
        return self.value['query_strategy']

    def is_classic(self):
        return self.value['classic']

    @classmethod
    def get_names(cls):
        return list(map(lambda x: x.value.get('name'), cls))

    @classmethod
    def from_name(cls, name):
        return next(filter(lambda x: x.value.get('name') == name, cls))

    @classmethod
    def get_query_strategy_from_name(cls, name):
        return AL_Strategy.from_name(cls, name).value.get('query_strategy')

    @classmethod
    def get_classics(cls):
        return list(filter(lambda x: x.value['classic'], cls))

    @classmethod
    def get_default(cls):
        return AL_Strategy.RDS


class Metrics(Enum):
    QUERYING_TIME = 'querying_time'
    CLASSIFICATION_TIME = 'classification_time'
    ACCURACY = 'accuracy'
    SELECTED_INDICES = 'selected_indices'

    @classmethod
    def get_metrics_dict(cls):
        return {metric: [] for metric in map(lambda m: m.value, cls)}


class Results():
    def __init__(self, strategy, classifier):
        self.results_dict = {'strategy': strategy, 'classifier': classifier}

    def append(self, num_labeled_samples, metric, value):
        if num_labeled_samples not in self.results_dict:
            self.results_dict[num_labeled_samples] = Metrics.get_metrics_dict()
        self.results_dict[num_labeled_samples][metric.value].append(value)

    def get_mean(self, num_labeled_samples, metric):
        return np.mean(self.results_dict[num_labeled_samples][metric.value])

    def save(self, filename):
        return np.save(filename, self.results_dict)


def run(al_params, n_splits=5):
    results = Results(al_params.strategy_name, al_params.classifier_name)

    for split in al_params.dataset.get_splits(n_splits):
        X_train, y_train, X_test, y_test, _, _ = split

        start_time = timer()
        idx, root_idx, X_initial, y_initial, X_pool, y_pool = get_samples(X_train, y_train,
                                                                          n_clusters=al_params.n_clusters,
                                                                          strategy=al_params.strategy)

        num_labeled_samples = len(X_initial)
        results.append(num_labeled_samples, Metrics.QUERYING_TIME, timer() - start_time)

        start_time = timer()
        learner = ActiveLearner(
            estimator=al_params.classifier,
            X_training=X_initial, y_training=y_initial,
            query_strategy=al_params.strategy.get_query_strategy())

        results.append(num_labeled_samples, Metrics.CLASSIFICATION_TIME, timer() - start_time)
        results.append(num_labeled_samples, Metrics.ACCURACY, learner.score(X_test, y_test))
        results.append(num_labeled_samples, Metrics.SELECTED_INDICES, root_idx.tolist())

        labeled_idx = np.empty(0, int)

        with trange(al_params.max_iterations) as pbar:
            for it in pbar:
                kwargs = dict()
                if not al_params.strategy.is_classic():
                    if al_params.n_instances > len(idx): break
                    kwargs = dict(idx=idx, labeled_idx=labeled_idx, y_root=y_initial)
                elif al_params.n_instances > len(X_pool):
                    break

                start_time = timer()

                query_idx, idx = learner.query(X_pool, n_instances=al_params.n_instances, **kwargs)

                if query_idx is None or len(query_idx) < al_params.n_instances: break

                num_labeled_samples += len(query_idx)
                results.append(num_labeled_samples, Metrics.QUERYING_TIME, timer() - start_time)

                start_time = timer()
                learner.teach(X=X_pool[query_idx], y=y_pool[query_idx])
                results.append(num_labeled_samples, Metrics.CLASSIFICATION_TIME, timer() - start_time)

                if al_params.strategy.is_classic():
                    X_pool = np.delete(X_pool, query_idx, axis=0)
                    y_pool = np.delete(y_pool, query_idx, axis=0)
                else:
                    labeled_idx = np.append(labeled_idx, query_idx)

                results.append(num_labeled_samples, Metrics.ACCURACY, learner.score(X_test, y_test))
                results.append(num_labeled_samples, Metrics.SELECTED_INDICES, query_idx.tolist())
                pbar.set_description(
                    (al_params.classifier_name + '  ' + al_params.strategy_name + ' %.2f') % results.get_mean(
                        num_labeled_samples, Metrics.ACCURACY))

    return results
