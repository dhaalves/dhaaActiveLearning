
# Installation<a name="installation"></a>
dhaaActiveLearning requires
- numpy
- scipy
- scikit-learn
- tqdm
- pandas
- googledrivedownloader
- modAL

You can install directly with pip:  
```
pip install git+https://github.com/dhaalves/dhaaActiveLearning.git
```


# Usage<a name="installation"></a>
First, you need a folder named 'datasets' which, for each dataset, it must contain 3 csv files (for features, labels and filenames) respectiong the following naming convention: 
- features: <dataset_name>_features.csv
- labels: <dataset_name>_labels.csv
- filenames: <dataset_name>_filenames.csv

You can check an example dataset under 'datasets' folder of this repository.

After that you run the follwing example (example.py)

```python
import numpy as np

import dhaaActiveLearning
from dhaaActiveLearning import AL_Strategy, AL_Parameters
from dhaaActiveLearning.classification import Classifier
from dhaaActiveLearning.dataset import Dataset

if __name__ == '__main__':
    np.random.seed(1) #for reproducibility on some al strategies

    print('AL Strategies:', AL_Strategy.get_names())
    print('Classifiers:', Classifier.get_names())
    print('Datasets:', Dataset.get_names())

    al_params = AL_Parameters(dataset_name='LEA-53', classifier_name='RF', strategy_name='MS', max_iterations=20)
    results = dhaaActiveLearning.run(al_params=al_params, n_splits=1)
    results.save('LEA-53-results')
```
