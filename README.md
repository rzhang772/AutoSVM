# MOE-SVM
A Python tool for automated SVM analysis with clustering and feature processing.

## For Quick Run

```bash

python src/run.py
```


## Features

### 1. Data Processing
- Support for LIBSVM format datasets
- Automatic sparse data handling
- L2 normalization
- Support for both classification and regression tasks

### 2. Clustering Methods (Optional)
- **K-means Clustering**
  - Automatic k selection using silhouette or GAP method
  - parallel computation (optional)
  - Support for fixed k value
  - TODO: select some features for clustering
- **Random Clustering**
  - Fixed k value required
  - Random assignment of data points
- **FIFO Clustering**
  - Fixed k value required
  - Sequential assignment of data points

### 3. Feature Processing (parallel computation optional)
- **Feature Independence Check selection**
  - Check the independence score of features
- **Feature Construction**
  - Standardization of original features
  - Feature discretization
  - Feature interactions
  - Statistical features
- **Feature Selection**
  - Mutual Information feature selection
  - Non-zero ratio filtering

### 4. Data Balancing (Optional)
- **Enable data balancing for classification tasks**
- **Minimum ratio required for each class (relative to majority class)**
- **Maximum samples to add per class**

### 5. SVM Training (parallel computation optional)
- Support for both LibSVM and ThunderSVM
- Classification and regression tasks
- Optional hyperparameter tuning with multiple optimizers:
  - Random Search
  - Grid Search
  - Bayesian Optimization
  - TPE (Tree-structured Parzen Estimators)

### 6. Evaluation Metrics
- **Classification Tasks**
  - Accuracy
  - F1 Score (binary/weighted)
- **Regression Tasks**
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - R² Score

## Usage

download the dataset from [here](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)
preprocess the dataset using `src/preprocessdata.py`

### Basic Command
```bash
python src/main.py --train <train_file> --test <test_file> \
--type <clf/reg> [options]
```


### Required Arguments
- `--train`: Path to training dataset
- `--test`: Path to test dataset
- `--type`: Task type ('clf' for classification, 'reg' for regression)

### Optional Arguments

#### Clustering Options
- `--clustering`: Enable clustering
- `--algorithm`: Clustering algorithm ('kmeans', 'random', 'fifo', default: 'kmeans')
- `--method`: Method for finding optimal k ('silhouette' or 'gap', default: 'silhouette')
- `--k`: Fixed number of clusters (required for random and fifo algorithms)
- `--parallel-cluster`: Enable parallel clustering computation
- `--cluster-jobs`: Number of parallel jobs for clustering (-1 for all CPUs)

#### Feature Processing Options
- `--feature-processing`: Enable feature processing
- `--feature-construction`: Enable feature construction
- `--mutual-info`: Enable mutual information feature selection
- `--qbsofs`: Enable QBSOFS feature selection
- `--parallel-feature`: Enable parallel feature processing
- `--feature-jobs`: Number of parallel jobs for feature processing (-1 for all CPUs)

#### Data Balancing Options
- `--balance-data`: Enable data balancing for classification tasks
- `--min-class-ratio`: Minimum ratio required for each class (relative to majority class)
- `--max-balance-samples`: Maximum samples to add per class

#### SVM Options
- `--svm-type`: SVM implementation ('libsvm' or 'thundersvm', default: 'libsvm')
- `--parallel-train`: Enable parallel SVM training
- `--train-jobs`: Number of parallel jobs for training (-1 for all CPUs)

#### Hyperparameter Tuning Options
- `--tune-hyperparams`: Enable hyperparameter tuning
- `--optimizer`: Optimization strategy ('random', 'grid', 'bayes', 'tpe', default: 'random')
- `--n-iter`: Number of iterations for hyperparameter tuning (default: 10)
- `--cv-folds`: Number of cross-validation folds (default: 3)
- `--parallel-tuning`: Enable parallel hyperparameter tuning
- `--tuning-jobs`: Number of parallel jobs for tuning (-1 for all CPUs)

### Examples

1. Basic Classification:
```bash
python src/main.py --train data.txt --test test.txt --type clf
```

2. Classification with Clustering:
```bash
python src/main.py --train data.txt --test test.txt --type clf \
--clustering --algorithm kmeans --method silhouette
```

3. Classification with Feature Processing:
```bash
python src/main.py --train data.txt --test test.txt --type clf \
--feature-processing --feature-construction --mutual-info --qbsofs
```

4. Classification with Hyperparameter Tuning:
```bash
python src/main.py --train data.txt --test test.txt --type clf \
--tune-hyperparams --optimizer bayes --n-iter 20
```

5. Full Pipeline:
```bash
python src/main.py --train data.txt --test test.txt --type clf \
--clustering --algorithm kmeans \
--feature-processing --mutual-info \
--tune-hyperparams --optimizer bayes
```

6. Classification with Parallel Training:
```bash
python src/main.py --train data.txt --test test.txt --type clf \
--clustering --algorithm kmeans \
--feature-processing --mutual-info \
--tune-hyperparams --optimizer bayes \
--parallel-train --train-jobs 4
```


7. Full Pipeline with Parallel Training:
```bash
python src/main.py --train data.txt --test test.txt --type clf \
--clustering --algorithm kmeans \
--feature-processing --mutual-info \
--tune-hyperparams --optimizer bayes \
--parallel-train --train-jobs -1
```

8. Classification with Data Balancing:
```bash
python src/main.py --train data.txt --test test.txt --type clf \
--balance-data --min-class-ratio 0.5 --max-balance-samples 100
```

9. Using Different Parallel Settings:
```bash
python src/main.py --train data.txt --test test.txt --type clf \
--clustering --algorithm kmeans \
--feature-processing --mutual-info \
--tune-hyperparams --optimizer bayes \
--parallel-train --train-jobs 4 --parallel-cluster --cluster-jobs 4
```

10. Full Pipeline with All Options:
```bash
python src/main.py --train data.txt --test test.txt --type clf \
--clustering --algorithm kmeans \
--feature-processing --mutual-info \
--tune-hyperparams --optimizer bayes \
--parallel-train --train-jobs -1 \
--balance-data --min-class-ratio 0.5 --max-balance-samples 100
```

## Project Structure
AutoSVM/
├── src/
│ ├── main.py
│ ├── dataset.py
│ ├── svmcluster.py
│ ├── feature_processor.py
│ ├── feature_constructor.py
│ ├── qbsofs.py
│ ├── svm_trainer.py
│ └── logger.py
├── tests/
│ ├── test_clustering.py
│ └── test_feature_processing.py
├── data/
│ └── processed/
│ ├── clf/
│ └── reg/
└── output/
  └── cluster/


## Requirements
- Python 3.7+
- NumPy
- SciPy
- scikit-learn
- ThunderSVM (optional)
- psutil

## for environment:
```bash
conda env create -f requirements.yml
conda activate autosvm
```

## for thundersvm:
```bash
pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install thundersvm==0.3.12
```

## for running:
```bash
python src/main.py --train ./data/processed/clf/aloi_train.txt --test ./data/processed/clf/aloi_test.txt --type clf \
--clustering --algorithm kmeans --balance-data --feature-processing --mutual-info \
--tune-hyperparams --optimizer bayes

python src/main.py --train ./data/processed/reg/cadata_train.txt --test ./data/processed/reg/cadata_test.txt --type reg \
--clustering --algorithm kmeans --balance-data --feature-processing --mutual-info \
--tune-hyperparams --optimizer bayes
```

```bash
python src/main.py --train ./data/processed/clf/aloi_train.txt --test ./data/processed/clf/aloi_test.txt --type clf \
--parallel-train  --clustering --algorithm kmeans --parallel-cluster \
--balance-data --feature-processing --mutual-info --parallel-feature \
--tune-hyperparams --optimizer bayes --parallel-tuning

python src/main.py --train ./data/processed/reg/cadata_train.txt --test ./data/processed/reg/cadata_test.txt --type reg \
--parallel-train  --clustering --algorithm kmeans --parallel-cluster \
--balance-data --feature-processing --mutual-info --parallel-feature \
--tune-hyperparams --optimizer bayes --parallel-tuning
```

python src/main.py --train ./data/processed/clf/HIGGS_train.txt --test ./data/processed/clf/HIGGS_test.txt --type clf --parallel-train  --clustering --algorithm kmeans --entropy-selection --cascade --feature-processing --mutual-info --parallel-cluster --balance-data --tune-hyperparams --optimizer bayes --parallel-tuning --log-level debug

python src/main.py --train ./data/processed/reg/cadata_train.txt --test ./data/processed/reg/cadata_test.txt --type reg --parallel-train  --clustering --algorithm kmeans --entropy-selection --cascade --feature-processing --mutual-info --parallel-cluster --balance-data --tune-hyperparams --optimizer bayes --parallel-tuning --log-level debug