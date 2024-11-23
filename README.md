# AutoSVM

A Python tool for automated SVM analysis with clustering and feature processing.

## Features

### 1. Data Processing
- Support for LIBSVM format datasets
- Automatic sparse data handling
- L2 normalization
- Support for both classification and regression tasks

### 2. Clustering Methods (Optional)
- **K-means Clustering**
  - Automatic k selection using silhouette or GAP method, (sample for large dataset)
  - Support for fixed k value
- **Random Clustering**
  - Fixed k value required
  - Random assignment of data points
- **FIFO Clustering**
  - Fixed k value required
  - Sequential assignment of data points

### 3. Feature Processing (Optional)
- **Feature Construction**
  - Standardization of original features
  - Feature discretization
  - Feature interactions
  - Statistical features
- **Feature Selection**
  - Mutual Information feature selection
  - QBSOFS (Quantum-Behaved Particle Swarm Optimization)
  - Non-zero ratio filtering
- **Parallel Feature Processing**
  - Enable parallel feature processing
  - Number of parallel jobs for feature processing (-1 for all CPUs)

### 4. Data Balancing (Optional)
- **Enable data balancing for classification tasks**
- **Minimum ratio required for each class (relative to majority class)**
- **Maximum samples to add per class**

### 5. SVM Training
- Support for both LibSVM and ThunderSVM
- Classification and regression tasks
- Sparse data optimization
- Individual model for each cluster
- Optional parallel training with multi-threading
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

### Basic Command
bash
python src/main.py --train <train_file> --test <test_file> \
--type <clf/reg> [options]


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
python src/main.py --train data.txt --test test.txt --type clf

2. Classification with Clustering:
python src/main.py --train data.txt --test test.txt --type clf \
--clustering --algorithm kmeans --method silhouette

3. Classification with Feature Processing:
python src/main.py --train data.txt --test test.txt --type clf \
--feature-processing --feature-construction --mutual-info --qbsofs

4. Classification with Hyperparameter Tuning:
python src/main.py --train data.txt --test test.txt --type clf \
--tune-hyperparams --optimizer bayes --n-iter 20

5. Full Pipeline:
python src/main.py --train data.txt --test test.txt --type clf \
--clustering --algorithm kmeans \
--feature-processing --mutual-info \
--tune-hyperparams --optimizer bayes

6. Classification with Parallel Training:
python src/main.py --train data.txt --test test.txt --type clf \
--clustering --algorithm kmeans \
--feature-processing --mutual-info \
--tune-hyperparams --optimizer bayes \
--parallel-train --train-jobs 4


7. Full Pipeline with Parallel Training:
python src/main.py --train data.txt --test test.txt --type clf \
--clustering --algorithm kmeans \
--feature-processing --mutual-info \
--tune-hyperparams --optimizer bayes \
--parallel-train --train-jobs -1

6. Classification with Data Balancing:
python src/main.py --train data.txt --test test.txt --type clf \
--balance-data --min-class-ratio 0.5 --max-balance-samples 100

7. Using Different Parallel Settings:
python src/main.py --train data.txt --test test.txt --type clf \
--clustering --algorithm kmeans \
--feature-processing --mutual-info \
--tune-hyperparams --optimizer bayes \
--parallel-train --train-jobs 4 --parallel-cluster --cluster-jobs 4

7. Full Pipeline with All Options:
python src/main.py --train data.txt --test test.txt --type clf \
--clustering --algorithm kmeans \
--feature-processing --mutual-info \
--tune-hyperparams --optimizer bayes \
--parallel-train --train-jobs -1 --balance-data --min-class-ratio 0.5 --max-balance-samples 100

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

python src/main.py --train ./data/processed/clf/aloi_train.txt --test ./data/processed/clf/aloi_test.txt --type clf --clustering --algorithm kmeans --balance-data --feature-processing --mutual-info --tune-hyperparams --optimizer bayes

python src/main.py --train ./data/processed/clf/aloi_train.txt --test ./data/processed/clf/aloi_test.txt --type clf --parallel-train  --clustering --algorithm kmeans --parallel-cluster --balance-data --feature-processing --mutual-info --parallel-feature --tune-hyperparams --optimizer bayes --parallel-tuning