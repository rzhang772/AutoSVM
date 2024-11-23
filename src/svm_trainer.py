import numpy as np
import scipy.sparse
from sklearn.svm import SVC, SVR
from thundersvm import SVC as ThunderSVC
from thundersvm import SVR as ThunderSVR
from typing import Dict, Any, Union, Tuple, List, Optional
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class SVMTrainer:
    """Class for training SVM models on processed clusters"""
    
    def __init__(self,
                 task_type: str,           # 'clf' or 'reg'
                 svm_type: str = 'libsvm', # 'libsvm' or 'thundersvm'
                 params_list: List[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize SVMTrainer
        
        Args:
            task_type: Type of task ('clf' or 'reg')
            svm_type: Type of SVM implementation ('libsvm' or 'thundersvm')
            params_list: List of SVM parameters for each cluster
            logger: Logger instance from main
        """
        self.task_type = task_type.lower()
        self.svm_type = svm_type.lower()
        self.params_list = params_list or []
        self.logger = logger or logging.getLogger('svm_trainer')
        
        # Validate task type
        if self.task_type not in ['clf', 'reg']:
            raise ValueError("Task type must be 'clf' or 'reg'")
            
        # Validate SVM type
        if self.svm_type not in ['libsvm', 'thundersvm']:
            raise ValueError("SVM type must be 'libsvm' or 'thundersvm'")
        
        # Initialize models dictionary
        self.models = {}
    
    def _get_model(self, params: Dict[str, Any]) -> Any:
        """Get appropriate SVM model based on task and implementation type"""
        if self.svm_type == 'libsvm':
            if self.task_type == 'clf':
                return SVC(**params)
            else:
                return SVR(**params)
        else:  # thundersvm
            if self.task_type == 'clf':
                return ThunderSVC(**params)
            else:
                return ThunderSVR(**params)
    
    def train_cluster_models(self,
                           processed_clusters: Dict[int, Tuple[scipy.sparse.csr_matrix, np.ndarray, list]],
                           use_sparse: bool = True) -> Dict[int, Dict]:
        """
        Train SVM models for each cluster
        
        Args:
            processed_clusters: Dictionary of processed cluster data
            use_sparse: Whether to use sparse format (if supported)
            
        Returns:
            Dictionary of training results for each cluster
        """
        results = {}
        
        for cluster_id, (X, y, selected_features) in processed_clusters.items():
            self.logger.debug(f"\nTraining SVM for cluster {cluster_id}")
            self.logger.debug(f"Data shape: {X.shape}")
            
            try:
                # Convert data format if needed
                if not use_sparse or self.svm_type == 'thundersvm':
                    X = X.toarray()
                    self.logger.info("Converted to dense format")
                
                # Get parameters for this cluster
                params = self.params_list[cluster_id]
                
                # Initialize and train model
                train_start = time.time()
                model = self._get_model(params)
                model.fit(X, y)
                train_time = time.time() - train_start
                
                # Store model and results
                self.models[cluster_id] = model
                results[cluster_id] = {
                    'training_time': train_time,
                    'n_support_vectors': model.n_support_[0] if self.task_type == 'clf' else model.support_.shape[0],
                    'selected_features': selected_features,
                    'task_type': self.task_type
                }
                
                # self.logger.info(f"Training completed in {train_time:.2f} seconds")
                # self.logger.info(f"Number of support vectors: {results[cluster_id]['n_support_vectors']}")
                # if self.task_type == 'reg':
                    # self.logger.info(f"Task type: Regression")
                # else:
                    # self.logger.info(f"Task type: Classification")
                
            except Exception as e:
                self.logger.error(f"Training failed for cluster {cluster_id}: {str(e)}")
                results[cluster_id] = {'error': str(e)}
        
        return results
    
    def predict(self,
               cluster_id: int,
               X: scipy.sparse.csr_matrix,
               use_sparse: bool = True) -> np.ndarray:
        """
        Make predictions using trained model for a specific cluster
        
        Args:
            cluster_id: Cluster identifier
            X: Test data
            use_sparse: Whether to use sparse format
            
        Returns:
            Predictions
        """
        if cluster_id not in self.models:
            raise ValueError(f"No trained model found for cluster {cluster_id}")
            
        try:
            if not use_sparse or self.svm_type == 'thundersvm':
                X = X.toarray()
            
            return self.models[cluster_id].predict(X)
            
        except Exception as e:
            self.logger.error(f"Prediction failed for cluster {cluster_id}: {str(e)}")
            raise
    
    def get_model(self, cluster_id: int) -> Any:
        """Get trained model for a specific cluster"""
        return self.models.get(cluster_id) 
    
    def _train_batch(self, batch_clusters, use_sparse: bool = True):
        """Train models for a batch of clusters"""
        batch_results = {}
        batch_models = {}
        
        for cluster_id, (X, y, selected_features) in batch_clusters.items():
            try:
                # Convert data format if needed
                if not use_sparse or self.svm_type == 'thundersvm':
                    X = X.toarray()
                
                # Get parameters for this cluster
                params = self.params_list[cluster_id]
                
                # Initialize and train model
                train_start = time.time()
                model = self._get_model(params)
                model.fit(X, y)
                train_time = time.time() - train_start
                
                # Store results
                batch_models[cluster_id] = model
                batch_results[cluster_id] = {
                    'training_time': train_time,
                    'n_support_vectors': model.n_support_[0] if self.task_type == 'clf' else model.support_.shape[0],
                    'selected_features': selected_features
                }
                
                # self.logger.info(f"Training completed in {train_time:.2f} seconds")
                # self.logger.info(f"Number of support vectors: {batch_results[cluster_id]['n_support_vectors']}")
                
            except Exception as e:
                batch_results[cluster_id] = {'error': str(e)}
                self.logger.error(f"Training failed for cluster {cluster_id}: {str(e)}")
        
        return batch_results, batch_models
    
    def train_cluster_models_parallel(self,
                                    processed_clusters: Dict[int, Tuple[scipy.sparse.csr_matrix, np.ndarray, list]],
                                    use_sparse: bool = True,
                                    n_jobs: int = -1) -> Dict[int, Dict]:
        """
        Train SVM models for each cluster in parallel
        
        Args:
            processed_clusters: Dictionary of processed cluster data
            use_sparse: Whether to use sparse format (if supported)
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            
        Returns:
            Dictionary of training results for each cluster
        """
        # Determine number of workers
        if n_jobs < 0:
            import multiprocessing
            n_jobs = min(multiprocessing.cpu_count(), len(processed_clusters))
        else:
            n_jobs = min(n_jobs, len(processed_clusters))
        
        self.logger.info(f"Starting parallel training with {n_jobs} workers")
        
        # Split clusters into batches
        cluster_items = list(processed_clusters.items())
        batch_size = max(1, len(cluster_items) // n_jobs)
        batches = [
            dict(cluster_items[i:i + batch_size])
            for i in range(0, len(cluster_items), batch_size)
        ]
        
        results = {}
        
        # Process batches in parallel
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_batch = {
                executor.submit(self._train_batch, batch, use_sparse): batch 
                for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    batch_results, batch_models = future.result()
                    results.update(batch_results)
                    self.models.update(batch_models)
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {str(e)}")
                    continue
        
        return results
    
    def _train_single_model(self,
                           X: scipy.sparse.csr_matrix,
                           y: np.ndarray,
                           selected_features: list,
                           cluster_id: int,
                           use_sparse: bool) -> Dict:
        """
        Train a single SVM model
        
        Args:
            X: Feature matrix
            y: Target labels
            selected_features: List of selected feature indices
            cluster_id: Cluster identifier
            use_sparse: Whether to use sparse format
            
        Returns:
            Dictionary containing training results
        """
        self.logger.debug(f"Training SVM for cluster {cluster_id}, Data shape: {X.shape}")
        
        try:
            # Convert data format if needed
            if not use_sparse or self.svm_type == 'thundersvm':
                X = X.toarray()
                self.logger.info("Converted to dense format")
            
            # Get parameters for this cluster
            params = self.params_list[cluster_id]
            
            # Initialize and train model
            train_start = time.time()
            model = self._get_model(params)
            model.fit(X, y)
            train_time = time.time() - train_start
            
            # Store model
            self.models[cluster_id] = model
            
            # Prepare results
            result = {
                'training_time': train_time,
                'n_support_vectors': model.n_support_[0] if self.task_type == 'clf' else model.support_.shape[0],
                'selected_features': selected_features
            }
            
            # self.logger.info(f"Training completed in {train_time:.2f} seconds")
            # self.logger.info(f"Number of support vectors: {result['n_support_vectors']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Training failed for cluster {cluster_id}: {str(e)}")
            raise
    
    def predict_parallel(self,
                        cluster_id: int,
                        X: scipy.sparse.csr_matrix,
                        use_sparse: bool = True,
                        n_jobs: int = -1) -> np.ndarray:
        """
        Make predictions using trained model for a specific cluster with parallel processing
        
        Args:
            cluster_id: Cluster identifier
            X: Test data
            use_sparse: Whether to use sparse format
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            
        Returns:
            Predictions
        """
        if cluster_id not in self.models:
            raise ValueError(f"No trained model found for cluster {cluster_id}")
        
        # Determine number of workers
        if n_jobs < 0:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        
        try:
            # Convert data format if needed
            if not use_sparse or self.svm_type == 'thundersvm':
                X = X.toarray()
            
            # Split data into chunks for parallel processing
            n_samples = X.shape[0]
            chunk_size = max(1, n_samples // n_jobs)
            chunks = [(i, min(i + chunk_size, n_samples)) 
                     for i in range(0, n_samples, chunk_size)]
            
            self.logger.info(f"Predicting with {len(chunks)} chunks using {n_jobs} workers")
            
            # Make predictions in parallel
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                future_to_chunk = {
                    executor.submit(
                        self._predict_chunk,
                        X[start:end],
                        cluster_id
                    ): (start, end) for start, end in chunks
                }
                
                # Collect results
                predictions = np.zeros(n_samples)
                for future in as_completed(future_to_chunk):
                    start, end = future_to_chunk[future]
                    try:
                        chunk_predictions = future.result()
                        predictions[start:end] = chunk_predictions
                    except Exception as e:
                        self.logger.error(f"Prediction failed for chunk {start}:{end}: {str(e)}")
                        raise
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Parallel prediction failed for cluster {cluster_id}: {str(e)}")
            raise
    
    def _predict_chunk(self,
                      X: Union[np.ndarray, scipy.sparse.csr_matrix],
                      cluster_id: int) -> np.ndarray:
        """
        Predict on a chunk of data
        
        Args:
            X: Chunk of test data
            cluster_id: Cluster identifier
            
        Returns:
            Predictions for the chunk
        """
        return self.models[cluster_id].predict(X)