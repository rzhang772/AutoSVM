import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.svm import SVC, SVR
from typing import Dict, Any, Optional, List
import logging
from scipy.stats import uniform, loguniform
from skopt import BayesSearchCV
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import time
import multiprocessing
import psutil

class SVMHyperparameterTuner:
    """Class for SVM hyperparameter tuning"""
    
    def __init__(self,
                 task_type: str,           
                 optimizer: str = 'random', 
                 n_iter: int = 10,         
                 cv: int = 3,              
                 random_state: int = 42):
        """
        Initialize hyperparameter tuner
        
        Args:
            task_type: Type of task ('clf' or 'reg')
            optimizer: Type of optimization strategy
            n_iter: Number of parameter settings sampled
            cv: Number of cross-validation folds
            random_state: Random seed
        """
        self.task_type = task_type.lower()
        self.optimizer = optimizer.lower()
        self.n_iter = n_iter
        self.cv_folds = cv
        self.random_state = random_state
        self.logger = logging.getLogger('main')
        
        # Validate optimizer type
        valid_optimizers = ['random', 'grid', 'bayes', 'tpe']
        if self.optimizer not in valid_optimizers:
            raise ValueError(f"Optimizer must be one of {valid_optimizers}")
        
        # Define parameter spaces for different optimizers
        self._define_parameter_spaces()
    
    def _define_parameter_spaces(self):
        """Define parameter spaces for different optimizers"""
        # Define common parameter ranges
        C_range = {
            'min': 1e-3,
            'max': 1e3,
            'n_points': 7  # for grid search
        }
        gamma_range = {
            'min': 1e-4,
            'max': 1e1,
            'n_points': 6  # for grid search
        }
        
        # Grid search space
        self.grid_space = {
            'C': np.logspace(np.log10(C_range['min']), np.log10(C_range['max']), C_range['n_points']),
            'gamma': np.logspace(np.log10(gamma_range['min']), np.log10(gamma_range['max']), gamma_range['n_points']),
            'kernel': ['rbf', 'linear']
        }
        
        # Random search space
        self.random_space = {
            'C': loguniform(1e-3, 1e3),
            'gamma': loguniform(1e-4, 1e1),
            'kernel': ['rbf', 'linear']
        }
        
        # Bayesian optimization space
        self.bayes_space = {
            'C': (1e-3, 1e3, 'log-uniform'),
            'gamma': (1e-4, 1e1, 'log-uniform'),
            'kernel': ['rbf', 'linear']
        }
        
        # TPE space
        self.tpe_space = {
            'C': hp.loguniform('C', np.log(1e-3), np.log(1e3)),
            'gamma': hp.loguniform('gamma', np.log(1e-4), np.log(1e1)),
            'kernel': hp.choice('kernel', ['rbf', 'linear'])
        }
        
        # Add epsilon parameter for regression
        if self.task_type == 'reg':
            self.grid_space['epsilon'] = np.linspace(0.1, 1.0, 10)
            self.random_space['epsilon'] = uniform(0, 1)
            self.bayes_space['epsilon'] = (0.0, 1.0, 'uniform')
            self.tpe_space['epsilon'] = hp.uniform('epsilon', 0, 1)
    
    def _get_min_samples_per_class(self, y: np.ndarray) -> int:
        """Get minimum number of samples for any class"""
        unique, counts = np.unique(y, return_counts=True)
        return min(counts)
    
    def calculate_cv_jobs(self, n_samples: int, tuning_jobs: int = None) -> int:
        """Calculate optimal number of parallel jobs for cross-validation
        
        Args:
            n_samples: Number of samples in the dataset
            tuning_jobs: Number of parallel tuning jobs (if using parallel tuning)
            
        Returns:
            optimal_jobs: Optimal number of parallel jobs for CV
        """
        cpu_count = multiprocessing.cpu_count()
        
        # For sequential tuning, use all available CPUs
        if tuning_jobs is None:
            return -1
        
        # For parallel tuning, calculate based on tuning_jobs
        if tuning_jobs >= cpu_count:
            return 1
        else:
            # Round to nearest integer
            return round(cpu_count / tuning_jobs) - 1
    
    def calculate_parallel_tuning_jobs(self, n_clusters: int) -> int:
        """Calculate optimal number of parallel jobs for tuning multiple clusters
        
        Args:
            n_clusters: Number of clusters to tune
            
        Returns:
            optimal_jobs: Optimal number of parallel tuning jobs based on system resources
        """
        cpu_count = multiprocessing.cpu_count()
        available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # Convert to GB
        
        # Estimate memory requirement per cluster tuning task (assume 4GB per cluster)
        memory_per_cluster = 4  # GB
        memory_based_jobs = int(available_memory / memory_per_cluster)
        
        # Consider number of clusters as maximum possible parallel jobs
        cluster_based_jobs = n_clusters
        
        # Take minimum of CPU, memory and cluster constraints
        optimal_jobs = max(1, min(cpu_count, memory_based_jobs, cluster_based_jobs))
        
        return optimal_jobs
    
    def _get_optimizer(self, estimator, n_samples: int):
        """Get appropriate optimizer based on configuration
        
        Args:
            estimator: Base estimator (SVC or SVR)
            n_samples: Number of samples for calculating optimal CV jobs
            
        Returns:
            optimizer: Configured optimizer instance with optimal parallel settings
        """
        # Calculate optimal number of CV jobs
        cv_jobs = self.calculate_cv_jobs(n_samples, getattr(self, 'tuning_jobs', None))
        
        self.logger.debug(f"Using {cv_jobs} parallel jobs for cross-validation")
        
        if self.optimizer == 'random':
            return RandomizedSearchCV(
                estimator=estimator,
                param_distributions=self.random_space,
                n_iter=self.n_iter,
                cv=self.cv,
                n_jobs=cv_jobs,
                random_state=self.random_state,
                verbose=0
            )
        elif self.optimizer == 'grid':
            self.logger.debug(f"GridSearchCV configuration: cv={self.cv}, n_jobs={cv_jobs}")
            return GridSearchCV(
                estimator=estimator,
                param_grid=self.grid_space,
                cv=self.cv,
                n_jobs=cv_jobs,
                verbose=0
            )
        elif self.optimizer == 'bayes':
            self.logger.debug(f"BayesSearchCV configuration: n_iter={self.n_iter}, cv={self.cv}, n_jobs={cv_jobs}")
            return BayesSearchCV(
                estimator=estimator,
                search_spaces=self.bayes_space,
                n_iter=self.n_iter,
                cv=self.cv,
                n_jobs=cv_jobs,
                random_state=self.random_state,
                verbose=0
            )
        else:  # TPE
            self.logger.debug(f"TPE configuration: max_evals={self.n_iter}, cv={self.cv}, n_jobs={cv_jobs}")
            return None
    
    def _objective(self, params, X, y, estimator, cv_jobs):
        """Objective function for TPE optimization"""
        try:
            # Convert kernel choice index to string for TPE
            if 'kernel' in params:
                kernel_choices = ['rbf', 'linear']
                params['kernel'] = kernel_choices[params['kernel']]
            
            # Log current parameters being evaluated
            self.logger.debug(f"Evaluating parameters: {params}")
            
            estimator.set_params(**params)
            start_time = time.time()
            scores = cross_val_score(estimator, X, y, cv=self.cv, n_jobs=cv_jobs)
            eval_time = time.time() - start_time
            
            mean_score = scores.mean()
            std_score = scores.std()
            
            self.logger.debug(f"Score: {mean_score:.4f} (±{std_score:.4f})")
            self.logger.debug(f"Evaluation time: {eval_time:.2f} seconds")
            
            return {'loss': -mean_score, 'status': STATUS_OK}
        except Exception as e:
            self.logger.error(f"Error in TPE objective function: {str(e)}")
            self.logger.error(f"Failed parameters: {params}")
            return {'loss': float('inf'), 'status': STATUS_OK}
    
    def tune(self, processed_clusters: Dict) -> List[Dict[str, Any]]:
        """
        Perform hyperparameter tuning for all clusters sequentially
        
        Args:
            processed_clusters: Dictionary of processed cluster data
            
        Returns:
            List of tuning results for each cluster
        """
        tuning_results = []
        
        for cluster_id, (X, y, _) in processed_clusters.items():
            # self.logger.info(f"Tuning hyperparameters for cluster {cluster_id}")
            
            # Validate input data
            if X.shape[0] < self.cv_folds:
                self.logger.warning(f"Dataset size ({X.shape[0]}) is smaller than cv_folds ({self.cv_folds})")
                self.cv_folds = max(2, X.shape[0] // 2)  # Ensure at least 2 folds
                self.logger.debug(f"Adjusted cv_folds to {self.cv_folds}")
            
            # Create base estimator
            if self.task_type == 'clf':
                estimator = SVC()
                
                # For classification, ensure minimum samples per class for cross-validation
                min_samples = self._get_min_samples_per_class(y)
                if min_samples < self.cv_folds:
                    self.cv_folds = max(2, min_samples)  # Ensure at least 2 folds
                    self.logger.debug(f"Adjusted cv_folds to {self.cv_folds} due to minimum class samples")
                
                self.cv = StratifiedKFold(
                    n_splits=self.cv_folds,
                    shuffle=True,
                    random_state=self.random_state
                )
            else:
                estimator = SVR()
                self.cv = KFold(
                    n_splits=self.cv_folds,
                    shuffle=True,
                    random_state=self.random_state
                )
            # self.logger.info(f"Tuning hyperparameters for cluster {cluster_id}...")
            # Perform optimization
            if self.optimizer == 'tpe':
                result = self._tune_tpe(X, y, estimator)
            else:
                result = self._tune_sklearn(X, y, estimator)
            
            tuning_results.append(result)
            self.logger.info(f"Best parameters for cluster {cluster_id}: {result['best_params']}")
            self.logger.debug(f"Best cross-validation score: {result['best_score']:.4f}")
        
        return tuning_results
    
    def _tune_batch(self, batch_clusters):
        """Tune hyperparameters for a batch of clusters"""
        batch_results = []
        
        for cluster_id, (X, y, _) in batch_clusters.items():
            # self.logger.info(f"\nTuning hyperparameters for cluster {cluster_id}")
            
            # Create base estimator and set up cross-validation
            if self.task_type == 'clf':
                estimator = SVC()
                min_samples = self._get_min_samples_per_class(y)
                cv_folds = max(2, min(self.cv_folds, min_samples))
                self.logger.debug(f"Adjusted cv_folds to {cv_folds} due to minimum class samples")
                self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            else:
                estimator = SVR()
                cv_folds = max(2, min(self.cv_folds, X.shape[0] // 2))
                self.logger.debug(f"Adjusted cv_folds to {cv_folds} due to dataset size")
                self.cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            # Perform optimization
            if self.optimizer == 'tpe':
                result = self._tune_tpe(X, y, estimator)
            else:
                result = self._tune_sklearn(X, y, estimator)
            
            batch_results.append(result)
            self.logger.info(f"Best parameters for cluster {cluster_id}: {result['best_params']}")
        
        return batch_results
    
    def tune_parallel(self,
                     processed_clusters: Dict,
                     n_jobs: int = -1) -> List[Dict[str, Any]]:
        """Tune hyperparameters for all clusters in parallel
        
        Args:
            processed_clusters: Dictionary of processed cluster data
            n_jobs: Number of parallel tuning tasks (-1 for auto-calculation)
        """
        cpu_count = multiprocessing.cpu_count()
        n_clusters = len(processed_clusters)
        
        # Calculate optimal number of parallel jobs if not specified
        if n_jobs < 0:
            n_jobs = min(cpu_count, n_clusters)
            self.logger.debug(f"Auto-calculated optimal number of parallel tuning jobs: {n_jobs}")
        else:
            original_n_jobs = n_jobs
            n_jobs = min(n_jobs, n_clusters)
            if original_n_jobs != n_jobs:
                self.logger.debug(f"Adjusted parallel tuning jobs from {original_n_jobs} to {n_jobs} based on cluster count")
        
        # Store tuning_jobs for use in _get_optimizer
        self.tuning_jobs = n_jobs
        
        self.logger.info(f"Starting parallel tuning with {n_jobs} workers")
        
        # Log system resources
        cpu_count = multiprocessing.cpu_count()
        available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
        self.logger.debug(f"System resources - CPU cores: {cpu_count}, Available memory: {available_memory:.2f}GB")
        
        # Split clusters into balanced batches for parallel processing
        cluster_items = list(processed_clusters.items())
        batch_size = max(1, len(cluster_items) // n_jobs)
        batches = [
            dict(cluster_items[i:i + batch_size])
            for i in range(0, len(cluster_items), batch_size)
        ]
        
        # Process batches in parallel using ProcessPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        all_results = []
        
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_to_batch = {
                executor.submit(self._tune_batch, batch): batch 
                for batch in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    self.logger.error(f"Batch tuning failed: {str(e)}")
                    continue
        
        return all_results
    
    def _tune_sklearn(self, X, y, estimator):
        """Tune hyperparameters using scikit-learn optimizers
        
        Args:
            X: Feature matrix
            y: Target labels
            estimator: Base estimator (SVC or SVR)
            
        Returns:
            Dictionary containing best parameters and score
        """
        # Pass the number of samples to _get_optimizer
        optimizer = self._get_optimizer(estimator, X.shape[0])
        optimizer.fit(X, y)
        
        return {
            'best_params': optimizer.best_params_,
            'best_score': optimizer.best_score_
        }
    
    def _tune_tpe(self, X, y, estimator):
        """Tune hyperparameters using Tree-structured Parzen Estimators (TPE)
        
        Args:
            X: Feature matrix
            y: Target labels
            estimator: Base estimator (SVC or SVR)
            
        Returns:
            Dictionary containing best parameters and score
        """
        # Calculate optimal CV jobs for TPE
        cv_jobs = self.calculate_cv_jobs(X.shape[0])
        
        trials = Trials()
        best = fmin(
            fn=lambda params: self._objective(params, X, y, estimator, cv_jobs),  # Pass cv_jobs to objective
            space=self.tpe_space,
            algo=tpe.suggest,
            max_evals=self.n_iter,
            trials=trials,
            rstate=np.random.RandomState(self.random_state)
        )
        
        # Convert kernel index to string
        kernel_choices = ['rbf', 'linear']
        best_params = {
            'C': float(np.exp(best['C'])),
            'gamma': float(np.exp(best['gamma'])),
            'kernel': kernel_choices[best['kernel']]
        }
        
        # Add epsilon for regression
        if self.task_type == 'reg':
            best_params['epsilon'] = float(best['epsilon'])
        
        # Get best score
        best_score = -min(trials.losses())  # Convert minimization to maximization
        
        return {
            'best_params': best_params,
            'best_score': best_score
        }