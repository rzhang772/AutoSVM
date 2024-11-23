import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.svm import SVC, SVR
from typing import Dict, Any, Optional, List
import logging
from scipy.stats import uniform, loguniform
from skopt import BayesSearchCV
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import time

class SVMHyperparameterTuner:
    """Class for SVM hyperparameter tuning"""
    
    def __init__(self,
                 task_type: str,           # 'clf' or 'reg'
                 optimizer: str = 'random', # 'random', 'grid', 'bayes', 'tpe'
                 n_iter: int = 10,         # Number of parameter settings sampled
                 cv: int = 3,              # Number of cross-validation folds
                 n_jobs: int = -1,         # Number of jobs for parallel processing
                 random_state: int = 42):
        """
        Initialize hyperparameter tuner
        
        Args:
            task_type: Type of task ('clf' or 'reg')
            optimizer: Type of optimization strategy
            n_iter: Number of parameter settings sampled
            cv: Number of cross-validation folds
            n_jobs: Number of jobs for parallel processing
            random_state: Random seed
        """
        self.task_type = task_type.lower()
        self.optimizer = optimizer.lower()
        self.n_iter = n_iter
        self.cv_folds = cv
        self.n_jobs = n_jobs
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
    
    def _get_optimizer(self, estimator):
        """Get appropriate optimizer based on configuration"""
        if self.optimizer == 'random':
            return RandomizedSearchCV(
                estimator=estimator,
                param_distributions=self.random_space,
                n_iter=self.n_iter,
                cv=self.cv,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=0
            )
        elif self.optimizer == 'grid':
            return GridSearchCV(
                estimator=estimator,
                param_grid=self.grid_space,
                cv=self.cv,
                n_jobs=self.n_jobs,
                verbose=0
            )
        elif self.optimizer == 'bayes':
            return BayesSearchCV(
                estimator=estimator,
                search_spaces=self.bayes_space,
                n_iter=self.n_iter,
                cv=self.cv,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=0
            )
        else:  # TPE
            return None  # TPE uses a different optimization approach
    
    def _objective(self, params, X, y, estimator):
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
            scores = cross_val_score(estimator, X, y, cv=self.cv, n_jobs=self.n_jobs)
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
            self.logger.info(f"\nTuning hyperparameters for cluster {cluster_id}")
            
            # Validate input data
            if X.shape[0] < self.cv_folds:
                self.logger.warning(f"Dataset size ({X.shape[0]}) is smaller than cv_folds ({self.cv_folds})")
                self.cv_folds = max(2, X.shape[0] // 2)  # Ensure at least 2 folds
                self.logger.info(f"Adjusted cv_folds to {self.cv_folds}")
            
            # Create base estimator
            if self.task_type == 'clf':
                estimator = SVC()
                
                # For classification, ensure minimum samples per class for cross-validation
                min_samples = self._get_min_samples_per_class(y)
                if min_samples < self.cv_folds:
                    self.cv_folds = max(2, min_samples)  # Ensure at least 2 folds
                    self.logger.info(f"Adjusted cv_folds to {self.cv_folds} due to minimum class samples")
                
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
            
            # Perform optimization
            if self.optimizer == 'tpe':
                result = self._tune_tpe(X, y, estimator)
            else:
                result = self._tune_sklearn(X, y, estimator)
            
            tuning_results.append(result)
            self.logger.info(f"Best parameters: {result['best_params']}")
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
                self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            else:
                estimator = SVR()
                cv_folds = max(2, min(self.cv_folds, X.shape[0] // 2))
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
        """
        Tune hyperparameters for all clusters in parallel
        
        Args:
            processed_clusters: Dictionary of processed cluster data
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            
        Returns:
            List of tuning results for each cluster
        """
        # Determine number of workers
        if n_jobs < 0:
            import multiprocessing
            n_jobs = min(multiprocessing.cpu_count(), len(processed_clusters))
        else:
            n_jobs = min(n_jobs, len(processed_clusters))
        
        self.logger.info(f"Starting parallel tuning with {n_jobs} workers")
        
        # Split clusters into batches
        cluster_items = list(processed_clusters.items())
        batch_size = max(1, len(cluster_items) // n_jobs)
        batches = [
            dict(cluster_items[i:i + batch_size])
            for i in range(0, len(cluster_items), batch_size)
        ]
        
        # Process batches in parallel
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        all_results = []
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_batch = {
                executor.submit(self._tune_batch, batch): batch 
                for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    self.logger.error(f"Batch tuning failed: {str(e)}")
                    continue
        
        return all_results
    
    def _tune_sklearn(self, X, y, estimator):
        """Tune hyperparameters using scikit-learn optimizers"""
        optimizer = self._get_optimizer(estimator)
        optimizer.fit(X, y)
        
        return {
            'best_params': optimizer.best_params_,
            'best_score': optimizer.best_score_
        }
    
    def _tune_tpe(self, X, y, estimator):
        """
        Tune hyperparameters using Tree-structured Parzen Estimators (TPE)
        
        Args:
            X: Feature matrix
            y: Target labels
            estimator: Base estimator (SVC or SVR)
            
        Returns:
            Dictionary containing best parameters and score
        """
        trials = Trials()
        best = fmin(
            fn=lambda params: self._objective(params, X, y, estimator),
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