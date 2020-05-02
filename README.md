# Implementations<a name="implementations"></a>
- AL Strategies: ['EN', 'MS', 'LC', 'EN-CLU', 'MS-CLU', 'LC-CLU', 'RBE', 'DBE', 'MST-BE', 'MST-CLU-DS', 'MST-CLU-DDE', 'RDS', 'MST-CLU-RDS', 'MST-CLU-RDS2']
- Classifiers: ['SVM', 'k-NN', 'RF', 'NB']
- Datasets: ['LEA-53'] 


# Installation<a name="installation"></a>
dhaaActiveLearning requires Python >= 3.5
- numpy
- scipy
- scikit-learn
- tqdm
- pandas
- googledrivedownloader
- modAL

You can install directly with pip:  
```
pip3 install git+https://github.com/dhaalves/dhaaActiveLearning.git
```


# Usage<a name="usage"></a>
First, you need a folder named 'datasets' which, for each dataset, must contain at least 2 CSV files (features, labels) respecting the following naming convention:
- features: '<dataset_name>_features.csv' **required**
- labels: '<dataset_name>_labels.csv' **required**
- filenames: '<dataset_name>_filenames.csv' **optional**

You can check an example dataset under 'datasets' folder of this repository.

After that, you can run the following example (example.py):

```python
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

    al_params = ALParams(dataset_name='LEA-53', classifier_name='RF', strategy_name='MS', max_iterations=20)
    results = dhaaActiveLearning.run(params=al_params, n_splits=1)
    results.save('LEA-53-results')
```
