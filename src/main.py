import argparse
from dataset import Dataset
from normalizer import DataNormalizer
from feature_constructor import FeatureConstructor
from svmcluster import SVMCluster, ClusterMethod
from logger import Logger
import numpy as np
import os
from sklearn.preprocessing import normalize
import scipy.sparse
import time
from datetime import datetime
from feature_processor import FeatureProcessor
from svm_trainer import SVMTrainer
import multiprocessing
import psutil
from feature_independence import FeatureIndependenceChecker
from hyperparameter_tuner import SVMHyperparameterTuner

# Configuration constants
K_MIN = 5
K_MAX = 10
ENABLE_SAMPLING = False  # Control whether to enable sampling
SAMPLE_RATIO = 0.1     # Sampling ratio

def normalize_sparse_data(X: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    """
    Normalize sparse data matrix using L2 normalization
    
    Args:
        X: Input sparse matrix
        
    Returns:
        L2 normalized sparse matrix
    """
    return normalize(X, norm='l2', axis=1, copy=True)


def main():
    # Start total time
    total_start_time = time.time()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run clustering analysis on SVM datasets')
    
    # SVM arguments
    parser.add_argument('--train', type=str, required=True,
                      help='Path to training dataset')
    parser.add_argument('--test', type=str, required=True,
                      help='Path to test dataset')
    parser.add_argument('--type', type=str, required=True,
                      choices=['clf', 'reg'],
                      help='Type of task (classification or regression)')
    parser.add_argument('--svm-type', type=str, default='libsvm',
                      choices=['libsvm', 'thundersvm'],
                      help='Type of SVM implementation')
    parser.add_argument('--parallel-train', action='store_true',
                      help='Enable parallel SVM training')
    parser.add_argument('--train-jobs', type=int, default=-1,
                      help='Number of parallel jobs for SVM training (-1 for all CPUs)')
    
    # Clustering arguments
    parser.add_argument('--clustering', action='store_true',
                      help='Enable clustering')
    parser.add_argument('--algorithm', type=str, default='kmeans',
                      choices=['kmeans', 'random', 'fifo'],
                      help='Clustering algorithm')
    parser.add_argument('--method', type=str, default='silhouette',
                      choices=['silhouette', 'gap'],
                      help='Method for finding optimal k')
    parser.add_argument('--k', type=int, default=None,
                      help='Fixed number of clusters (optional)')
    parser.add_argument('--parallel-cluster', action='store_true',
                      help='Enable parallel clustering')
    parser.add_argument('--cluster-jobs', type=int, default=-1,
                      help='Number of parallel jobs for clustering (-1 for all CPUs)')
    
    # Data balancing arguments
    parser.add_argument('--balance-data', action='store_true',
                      help='Enable data balancing for classification tasks')
    parser.add_argument('--min-class-ratio', type=float, default=0.1,
                      help='Minimum ratio required for each class')
    parser.add_argument('--max-balance-samples', type=int, default=100,
                      help='Maximum samples to add per class')

    # Feature processing arguments
    parser.add_argument('--feature-processing', action='store_true',
                      help='Enable feature processing')
    parser.add_argument('--independence-check', action='store_true',
                      help='Enable feature independence check')
    parser.add_argument('--independence-threshold', type=float, default=1e-4,
                      help='Threshold for feature independence check')
    parser.add_argument('--independence-smooth', type=float, default=1.0,
                      help='Smoothing factor for independence check')
    parser.add_argument('--feature-construction', action='store_true',
                      help='Enable feature construction')
    parser.add_argument('--discretize-ratio', type=float, default=0.1,
                      help='Ratio of features to discretize (default: 0.3)')
    parser.add_argument('--mutual-info', action='store_true',
                      help='Enable mutual information feature selection')
    parser.add_argument('--qbsofs', action='store_true',
                      help='Enable QBSOFS feature selection')
    parser.add_argument('--parallel-feature', action='store_true',
                      help='Enable parallel feature processing')
    parser.add_argument('--feature-jobs', type=int, default=-1,
                      help='Number of parallel jobs for feature processing (-1 for all CPUs)')
    
    
    # Hyperparameter tuning arguments
    parser.add_argument('--tune-hyperparams', action='store_true',
                      help='Enable SVM hyperparameter tuning')
    parser.add_argument('--optimizer', type=str, default='random',
                      choices=['random', 'grid', 'bayes', 'tpe'],
                      help='Hyperparameter optimization strategy')
    parser.add_argument('--n-iter', type=int, default=10,
                      help='Number of parameter settings sampled')
    parser.add_argument('--cv-folds', type=int, default=3,
                      help='Number of cross-validation folds')
    parser.add_argument('--parallel-tuning', action='store_true',
                      help='Enable parallel hyperparameter tuning')
    parser.add_argument('--tuning-jobs', type=int, default=-1,
                      help='Number of parallel jobs for hyperparameter tuning (-1 for all CPUs)')
    
    # Add logging level argument
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['debug', 'info', 'warning', 'error', 'critical'],
                      help='Set the logging level (case-insensitive)')
    
    args = parser.parse_args()
    
    # Get dataset name for log file
    train_dataset_name = os.path.splitext(os.path.basename(args.train))[0]
    
    # Create log directory if it doesn't exist
    log_dir = "./log"
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize logger with dataset name in log file and set log level
    # Convert log level to uppercase for logging module
    logger = Logger().get_logger(
        'main', 
        filename=os.path.join(
            log_dir, 
            f'log_{train_dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        ),
        level=args.log_level.upper()  # Convert to uppercase before passing to logger
    )
    
    # Log the current logging level
    logger.debug(f"Logging level set to: {args.log_level}")
    
    logger.info(f"Started clustering analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Log execution configuration
    logger.info("\n=== Execution Configuration ===")
    logger.info("Datasets:")
    logger.info(f"  Training data: {args.train}")
    logger.info(f"  Test data: {args.test}")
    
    logger.info("Task Configuration:")
    logger.info(f"  Task type: {args.type}")
    logger.info(f"  SVM implementation: {args.svm_type}")
    logger.info(f"  Parallel training: {args.parallel_train}")
    if args.parallel_train:
        logger.info(f"  Training jobs: {args.train_jobs}")
    
    logger.info("Clustering Configuration:")
    logger.info(f"  Clustering enabled: {args.clustering}")
    if args.clustering:
        logger.info(f"  Algorithm: {args.algorithm}")
        if args.algorithm == 'kmeans':
            logger.info(f"  Method: {args.method}")
            if args.k:
                logger.info(f"  Fixed k: {args.k}")
            else:
                logger.info(f"  k range: {K_MIN} to {K_MAX}")
        else:
            logger.info(f"  Fixed k: {args.k}")
        logger.info(f"  Parallel clustering: {args.parallel_cluster}")
        if args.parallel_cluster:
            logger.info(f"  Cluster jobs: {args.cluster_jobs}")
        logger.info(f"  Sampling enabled: {ENABLE_SAMPLING}")
        logger.info(f"  Sampling ratio: {SAMPLE_RATIO}")

    logger.info("Data Balancing Configuration:")
    logger.info(f"  Data balancing enabled: {args.balance_data}")
    if args.balance_data:
        logger.info(f"  Minimum class ratio: {args.min_class_ratio}")
        logger.info(f"  Maximum balance samples: {args.max_balance_samples}")
        
    logger.info("Feature Processing Configuration:")
    logger.info(f"  Feature processing enabled: {args.feature_processing}")
    if args.feature_processing:
        logger.info(f"  Independence check: {args.independence_check}")
        logger.info(f"  Independence threshold: {args.independence_threshold}")
        logger.info(f"  Independence smoothing: {args.independence_smooth}")
        logger.info(f"  Feature construction: {args.feature_construction}")
        logger.info(f"  Mutual info selection: {args.mutual_info}")
        logger.info(f"  QBSOFS selection: {args.qbsofs}")
    
    logger.info("Hyperparameter Tuning Configuration:")
    logger.info(f"  Tuning enabled: {args.tune_hyperparams}")
    if args.tune_hyperparams:
        logger.info(f"  Optimizer: {args.optimizer}")
        logger.info(f"  Number of iterations: {args.n_iter}")
        logger.info(f"  Cross-validation folds: {args.cv_folds}")
        logger.info(f"  Parallel tuning: {args.parallel_tuning}")
        if args.parallel_tuning:
            logger.info(f"  Tuning jobs: {args.tuning_jobs}")
    
    
    
    logger.info("\n" + "="*50 + "\n")
    
    # Validate algorithm and method combination
    if args.algorithm != 'kmeans' and args.method != 'silhouette':
        logger.error(f"Method '{args.method}' can only be used with kmeans algorithm")
        return 1
    
    # Validate algorithm choice for optimal k search
    if args.k is None and args.algorithm != 'kmeans':
        logger.error(f"Algorithm '{args.algorithm}' requires a fixed k value")
        return 1
    
    try:
        # Load training data
        logger.info("Loading training data...")
        load_train_start = time.time()
        train_dataset = Dataset(args.train)
        X_train, y_train = train_dataset.load_data()
        load_train_time = time.time() - load_train_start
        logger.info(f"Training data loaded in {load_train_time:.2f} seconds")
        
        # Load test data
        logger.info("Loading test data...")
        load_test_start = time.time()
        test_dataset = Dataset(args.test)
        X_test, y_test = test_dataset.load_data()
        load_test_time = time.time() - load_test_start
        logger.info(f"Test data loaded in {load_test_time:.2f} seconds")

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        # Verify feature dimensions match
        if X_train.shape[1] != X_test.shape[1]:
            logger.error("Feature dimensions do not match between train and test sets")
            return 1
        
        # Normalize data
        logger.info("Normalizing training data using L2 normalization...")
        norm_start = time.time()
        normalizer = DataNormalizer(norm='l2', logger=logger)
        X_train_normalized = normalizer.fit_transform(X_train)
        
        norm_time = time.time() - norm_start
        logger.info(f"Data normalization completed in {norm_time:.2f} seconds")
        
        # Perform clustering if enabled
        if args.clustering:
            logger.info("Starting clustering...")
            cluster_start = time.time()
            # Initialize clusterer with main logger
            clusterer = SVMCluster(
                sample_ratio=SAMPLE_RATIO,
                enable_sampling=ENABLE_SAMPLING,
                logger=logger  # Pass the main logger
            )
            best_k, results, labels, _ = clusterer.fit_predict(
                X_train_normalized,
                k=args.k,
                k_range=range(K_MIN, K_MAX + 1) if args.k is None else None,
                method=args.method,
                algorithm=args.algorithm,
                parallel=args.parallel_cluster,  # Use clustering-specific parallel parameter
                n_jobs=args.cluster_jobs       # Use clustering-specific jobs parameter
            )
            cluster_time = time.time() - cluster_start
            logger.info(f"Clustering completed in {cluster_time:.2f} seconds")
            
            # Log cluster sizes
            unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
            logger.debug("\nCluster size distribution:")
            for label, size in zip(unique_labels, cluster_sizes):
                logger.debug(f"Cluster {label}: {size} samples ({size/len(labels)*100:.2f}%)")
            
            # Save scores for k results
            if args.clustering and args.algorithm == 'kmeans' and args.k is None:
                save_start = time.time()
                clusterer.save_results(
                    os.path.basename(args.train),
                    best_k,
                    results,
                    args.method
                )
                save_time = time.time() - save_start
                logger.info(f"Results for k scores saved in {save_time:.2f} seconds")
            
            # Validate clustering results
            if len(np.unique(labels)) != best_k:
                logger.warning(f"Actual number of clusters {len(np.unique(labels))} " 
                             f"differs from expected {best_k}")
        else:
            # If clustering is not enabled, use the entire dataset as a single cluster
            labels = np.zeros(X_train_normalized.shape[0], dtype=int)
            best_k = 1
            logger.info("Clustering is disabled. Using entire dataset as a single cluster.")
        
        # Data balancing for classification task (only for training data)
        logger.info("\n=== Data Balancing ===")
        if args.type == 'reg':
            logger.info("Data balancing is not required for regression tasks")

        if args.type == 'clf' and args.clustering and args.algorithm == 'kmeans' and args.balance_data:
            balance_start_time = time.time()  # Start timing
            
            from data_balancer import DataBalancer
            
            balancer = DataBalancer(
                min_ratio=args.min_class_ratio,
                max_samples=args.max_balance_samples,
                random_state=42,
                logger=logger
            )
            
            # Create initial clusters dictionary for balancing
            initial_clusters = {}
            unique_labels = np.unique(labels)
            for cluster_id in unique_labels:
                cluster_mask = labels == cluster_id
                X_cluster = X_train_normalized[cluster_mask]
                y_cluster = y_train[cluster_mask]
                initial_clusters[cluster_id] = (X_cluster, y_cluster, list(range(X_cluster.shape[1])))
            
            # Balance clusters
            balanced_clusters = balancer.balance_clusters(
                X_train_normalized,
                y_train,
                initial_clusters
            )
            
            balance_time = time.time() - balance_start_time  # Calculate time
            logger.info(f"\nData balancing completed in {balance_time:.2f} seconds")
        else:
            balance_time = 0  # Set to 0 if not used===================================================================
            balanced_clusters = {
                cluster_id: (X_train_normalized[labels == cluster_id],
                           y_train[labels == cluster_id],
                           list(range(X_train_normalized.shape[1])))
                for cluster_id in np.unique(labels)
            }
        
        # Feature processing
        if args.feature_processing:
            logger.info("=== Feature Processing ===")
            feature_process_start = time.time()

            # Feature independence check
            independence_start = time.time()
            if args.independence_check:
                logger.info("=== Feature Independence Check ===")
                checker = FeatureIndependenceChecker(
                    threshold=args.independence_threshold,
                    smooth_factor=args.independence_smooth,
                    n_jobs=-1,
                    logger=logger
                )
                
                balanced_clusters = checker.fit_transform(
                    balanced_clusters,
                    args.train  # Pass the dataset name
                )
                independence_time = time.time() - independence_start
                logger.info(f"Feature independence check completed in {independence_time:.2f} seconds")
            
            # Feature construction
            construction_start = time.time()
            if args.feature_construction:
                try:
                    logger.info("=== Feature Construction ===")
                    constructor = FeatureConstructor(n_bins=5, discretize_ratio=args.discretize_ratio, logger=logger)
                    balanced_clusters = constructor.fit_transform(balanced_clusters)
                    feature_construction_time = time.time() - construction_start
                    logger.info(f"Feature construction completed in {feature_construction_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"Feature construction failed: {str(e)}")
        

            # Initialize feature processor
            mutual_info_qbsofs_start = time.time()
            logger.info("=== Mutual Info and QBSOFS ===")
            feature_processor = FeatureProcessor(
                task_type=args.type,
                enable_mutual_info=args.mutual_info,
                enable_qbsofs=args.qbsofs,
                non_zero_threshold=0.01,
                min_features=5,
                qbsofs_params={
                    'n_particles': 20,
                    'max_iter': 100,
                    'alpha': 0.7,
                    'beta': 0.5,
                    'n_folds': 5,
                    'random_state': 42
                },
                logger=logger
            )
            
            # Process features for each balanced cluster
            if args.parallel_feature:
                processed_clusters = feature_processor.fit_transform_parallel(
                    balanced_clusters,
                    n_jobs=args.feature_jobs
                )
            else:
                processed_clusters = feature_processor.fit_transform(
                    balanced_clusters
                )
            mutual_info_qbsofs_time = time.time() - mutual_info_qbsofs_start
            feature_process_time = time.time() - feature_process_start
            logger.info(f"Mutual info and QBSOFS completed in {mutual_info_qbsofs_time:.2f} seconds")
            logger.info(f"\nFeature processing completed in {feature_process_time:.2f} seconds")
        else:
            processed_clusters = balanced_clusters
        
        # SVM Training and Testing
        logger.info("\n=== SVM Training and Testing ===")
        svm_start_time = time.time()

        # Initialize base parameters
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'cache_size': 2000,
            'verbose': False
        }
        
        # Initialize parameters list for each cluster
        params_list = []
        
        # Perform hyperparameter tuning if enabled
        if args.tune_hyperparams:
            logger.info("\n=== Hyperparameter Tuning ===")
            tuning_start_time = time.time()
            
            # Initialize hyperparameter tuner with basic configuration
            # The tuner will handle optimal parallelization internally
            tuner = SVMHyperparameterTuner(
                task_type=args.type,
                optimizer=args.optimizer,
                n_iter=args.n_iter,
                cv=args.cv_folds,
                random_state=42
            )
            
            try:
                if args.parallel_tuning:
                    # Perform parallel tuning across clusters
                    # n_jobs will be calculated automatically if set to -1
                    tuning_results = tuner.tune_parallel(
                        processed_clusters,
                        n_jobs=args.tuning_jobs
                    )
                else:
                    # Perform sequential tuning for each cluster
                    tuning_results = tuner.tune(processed_clusters)
                
                # Store tuning results and combine with default parameters
                if tuning_results:
                    params_list = [
                        {**default_params, **result['best_params']}
                        for result in tuning_results
                    ]
                else:
                    logger.warning("Hyperparameter tuning failed, using default parameters")
                    params_list = [default_params] * len(processed_clusters)
            except Exception as e:
                logger.error(f"Hyperparameter tuning failed: {str(e)}")
                logger.warning("Using default parameters")
                params_list = [default_params] * len(processed_clusters)
            
            tuning_time = time.time() - tuning_start_time
            logger.info(f"\nHyperparameter tuning completed in {tuning_time:.2f} seconds")
        else:
            tuning_time = 0  # Set to 0 if not used
            # Use default parameters for all clusters
            for _ in processed_clusters:
                params_list.append(default_params)
        
        # Initialize SVM trainer with appropriate parameters
        svm_trainer = SVMTrainer(
            task_type=args.type,
            svm_type=args.svm_type,
            params_list=params_list,
            logger=logger  # Pass main logger
        )
        
        # Train models for each cluster
        logger.info("\n=== SVM Training ===")
        train_start_time = time.time()
        
        if args.parallel_train:
            logger.info("Using parallel training")
            training_results = svm_trainer.train_cluster_models_parallel(
                processed_clusters,
                use_sparse=(args.svm_type == 'libsvm'),
                n_jobs=args.train_jobs
            )
        else:
            logger.info("Using sequential training")
            training_results = svm_trainer.train_cluster_models(
                processed_clusters,
                use_sparse=(args.svm_type == 'libsvm')
            )
        
        train_time = time.time() - train_start_time
        
        # Log training results
        for cluster_id, result in training_results.items():
            X_cluster, y_cluster, _ = processed_clusters[cluster_id]
            logger.info(f"Cluster {cluster_id} Training:")
            logger.debug(f"Training samples: {X_cluster.shape[0]}")
            logger.debug(f"Features: {X_cluster.shape[1]}")
            if 'error' in result:
                logger.error(f"Training failed: {result['error']}")
                continue
            logger.info(f"Training time: {result['training_time']:.2f} seconds")
            logger.info(f"Support vectors: {result['n_support_vectors']}")
        
        logger.info(f"\nTotal SVM training time: {train_time:.2f} seconds")
        
        # Make predictions and evaluate
        logger.info("\n=== SVM Testing ===")
        test_start_time = time.time()

        # Normalize test data
        logger.info("Normalizing test data...")
        X_test_normalized = normalizer.transform(X_test)
        
        # Cluster test data
        logger.info("Clustering test data...")
        cluster_assign_start = time.time()
        if args.clustering:
            test_labels = clusterer.predict(X_test_normalized)
        else:
            # If clustering is disabled, assign all test data to cluster 0
            test_labels = np.zeros(X_test_normalized.shape[0], dtype=int)
            logger.info("Clustering disabled, all test data assigned to single cluster")
            
        test_clusters = {
                cluster_id: (X_test_normalized[test_labels == cluster_id],
                           y_test[test_labels == cluster_id],
                           list(range(X_test_normalized.shape[1])))
                for cluster_id in np.unique(test_labels)
            }
        cluster_assign_time = time.time() - cluster_assign_start
        logger.info(f"Assignment time: {cluster_assign_time:.2f} seconds")
        
        # no need balance test data
        if args.independence_check:
            # independence check is not needed for test data
            logger.info("Independence Checking test data...")
            test_clusters = checker.transform(test_clusters)
        
        if args.feature_construction:
            # feature construction is not needed for test data
            logger.info("Feature Construction test data...")
            test_clusters = constructor.transform(test_clusters)
        
        if args.feature_processing:
            # other feature processing is not needed for test data
            logger.info("Mutual and qbsofs test data...")
            test_clusters = feature_processor.transform(test_clusters)

        # predict and evaluate
        logger.info("predicting...")
        predict_start_time = time.time()
        if args.parallel_train:
            predictions = svm_trainer.predict_parallel(test_clusters)
        else:
            predictions = svm_trainer.predict(test_clusters)   
        predict_time = time.time() - predict_start_time
        test_time = time.time() - test_start_time
        logger.info(f"Prediction time: {predict_time:.2f} seconds")
        logger.info(f"Testing time: {test_time:.2f} seconds")

        # merge predictions and true labels
        if(len(predictions) != len(test_clusters)):
            logger.error("Predictions and test clusters have different lengths")
            return 1
        
        final_results = {} # {cluster_id: (prediction:list, true_labels:list)}
        for cluster_id, prediction in predictions.items():
            if(cluster_id in test_clusters):
                final_results[cluster_id] = (prediction, test_clusters[cluster_id][1])
            else:
                logger.error(f"Cluster {cluster_id} not found in test clusters")
                return 1

        # evaluate
        all_predictions = []
        all_true_labels = []
        total_test_samples = 0
        cluster_metrics = {}
        for cluster_id, (prediction, true_labels) in final_results.items():
            all_predictions.extend(prediction)
            all_true_labels.extend(true_labels)
            total_test_samples += len(prediction)  

            # Calculate metrics and log results
            if args.type == 'clf':
                from sklearn.metrics import accuracy_score, f1_score
                accuracy = accuracy_score(true_labels, prediction)
                if len(np.unique(true_labels)) == 2:
                    f1 = f1_score(true_labels, prediction, average='binary')
                else:
                    f1 = f1_score(true_labels, prediction, average='weighted')
                logger.info(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
                cluster_metrics[cluster_id] = {'accuracy': accuracy, 'f1': f1}
                logger.info(f"Cluster {cluster_id} Test samples: {len(prediction)}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
            else:
                from sklearn.metrics import mean_squared_error, r2_score
                mse = mean_squared_error(true_labels, prediction)
                rmse = np.sqrt(mse)
                r2 = r2_score(true_labels, prediction)
                logger.info(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
                cluster_metrics[cluster_id] = {'mse': mse, 'rmse': rmse, 'r2': r2}
                logger.info(f"Cluster {cluster_id} Test samples: {len(prediction)}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        svm_time = time.time() - svm_start_time
        total_time = time.time() - total_start_time

        # Calculate and log overall metrics
        logger.info("=== Overall Test Results ===")
        logger.info(f"Total test samples: {total_test_samples}")
        if args.type == 'clf':
            overall_accuracy = accuracy_score(all_true_labels, all_predictions)
            if len(np.unique(all_true_labels)) == 2:
                overall_f1 = f1_score(all_true_labels, all_predictions, average='binary')
            else:
                overall_f1 = f1_score(all_true_labels, all_predictions, average='weighted')
            logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
            logger.info(f"Overall F1 Score: {overall_f1:.4f}")
        else:  # regression task
            overall_mse = mean_squared_error(all_true_labels, all_predictions)
            overall_rmse = np.sqrt(overall_mse)
            overall_r2 = r2_score(all_true_labels, all_predictions)
            
            logger.info(f"Overall MSE: {overall_mse:.4f}")
            logger.info(f"Overall RMSE: {overall_rmse:.4f}")
            logger.info(f"Overall R² Score: {overall_r2:.4f}")
        
        # Update timing summary
        logger.info("=== Timing Summary ===")
        logger.info(f"Loading training data: {load_train_time:.2f} seconds")
        logger.info(f"Loading test data: {load_test_time:.2f} seconds")
        logger.info(f"Data normalization: {norm_time:.2f} seconds")
        if args.clustering:
            logger.info(f"Clustering: {cluster_time:.2f} seconds")
        if args.balance_data:
            logger.info(f"Data balancing: {balance_time:.2f} seconds")
        if args.independence_check:
            logger.info(f"Independence checking: {independence_time:.2f} seconds")
        if args.feature_construction:
            logger.info(f"Feature construction: {feature_construction_time:.2f} seconds")
        if args.feature_processing:
            logger.info(f"Feature processing: {feature_process_time:.2f} seconds")
        
        logger.info(f"SVM processing: {svm_time:.2f} seconds")
        if args.tune_hyperparams:
            logger.info(f"  Hyperparameter tuning: {tuning_time:.2f} seconds")
        logger.info(f"  Training time: {train_time:.2f} seconds")
        logger.info(f"  Testing time: {test_time:.2f} seconds")
        logger.info(f"      Prediction time: {predict_time:.2f} seconds")
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
    except ValueError as ve:
        logger.error(f"Parameter error: {str(ve)}")
        return 1
    except MemoryError:
        logger.error("Memory insufficient, please try reducing data size or enabling sampling")
        return 1
    except Exception as e:
        logger.error(f"Unknown error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 