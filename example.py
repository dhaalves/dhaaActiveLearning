import numpy as np

import dhaaActiveLearning
from dhaaActiveLearning import ALStrategy, ALParams
from dhaaActiveLearning.classification import Classifier
from dhaaActiveLearning.dataset import Dataset

if __name__ == '__main__':
    np.random.seed(1) #for reproducibility on some al strategies

    print('AL Strategies:', ALStrategy.get_names())
    print('Classifiers:', Classifier.get_names())
    print('Datasets:', Dataset.get_names())

    params = ALParams(dataset_name='LEA-53', classifier_name='RF', strategy_name='RDS', max_iterations=5)
    results = dhaaActiveLearning.run(params=params, n_splits=1)
    print(results.get_acc())
    # results.save('LEA-53-results')
    # print(np.load('LEA-53-results.npy'))