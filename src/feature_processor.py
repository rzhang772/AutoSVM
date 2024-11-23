import numpy as np
import scipy.sparse
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, SelectPercentile
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict
from qbsofs import QBSOFS
from feature_constructor import FeatureConstructor

class FeatureProcessor:
    """Class for processing features in clustered subsets"""
    
    def __init__(self, 
                 task_type: str,           # 'clf' or 'reg'
                 enable_construction: bool = False,
                 enable_mutual_info: bool = True,
                 enable_qbsofs: bool = True,
                 non_zero_threshold: float = 0.01,
                 min_features: int = 10,
                 qbsofs_params: Dict = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize FeatureProcessor
        
        Args:
            task_type: Type of task ('clf' or 'reg')
            enable_construction: Whether to enable feature construction
            enable_mutual_info: Whether to enable mutual information feature selection
            enable_qbsofs: Whether to enable QBSOFS feature selection
            non_zero_threshold: Minimum ratio of non-zero values required
            min_features: Minimum number of features to keep
            qbsofs_params: Parameters for QBSOFS
            logger: Logger instance from main
        """
        self.task_type = task_type.lower()
        self.enable_construction = enable_construction
        self.enable_mutual_info = enable_mutual_info
        self.enable_qbsofs = enable_qbsofs
        self.non_zero_threshold = non_zero_threshold
        self.min_features = min_features
        
        # Initialize QBSOFS if enabled
        if self.enable_qbsofs:
            self.qbsofs = QBSOFS(**(qbsofs_params or {}))
        
        # Use provided logger or create new one
        self.logger = logger
        self.feature_stats = {}
        
        if self.enable_construction:
            self.constructor = FeatureConstructor(n_bins=5)
        
    
    def _process_cluster_features(self, 
                                X: scipy.sparse.csr_matrix,
                                y: np.ndarray,
                                cluster_id: int) -> Tuple[scipy.sparse.csr_matrix, List[int]]:
        """Process features for a single cluster"""
        # Feature construction
        if self.enable_construction:
            try:
                X_augmented, feature_descriptions = self.constructor.construct_features(X)
                self.logger.info(f"Original features: {X.shape[1]}")
                self.logger.info(f"After construction: {X_augmented.shape[1]}")
                X = X_augmented
            except Exception as e:
                self.logger.error(f"Feature construction failed: {str(e)}")
        
        # Initial non-zero filtering
        n_samples = X.shape[0]
        non_zero_counts = np.array((X != 0).sum(axis=0)).flatten()
        non_zero_ratios = non_zero_counts / n_samples
        
        non_zero_mask = non_zero_ratios >= self.non_zero_threshold
        if not any(non_zero_mask):
            self.logger.warning(f"No features passed non-zero threshold {self.non_zero_threshold}")
            return None, []
            
        X_filtered = X[:, non_zero_mask]
        initial_features = np.where(non_zero_mask)[0]
        
        self.logger.debug(f"Features after non-zero filtering: {len(initial_features)}")
        
        # Mutual Information feature selection
        if self.enable_mutual_info:
            try:
                # Choose scoring function based on task type
                score_func = mutual_info_classif if self.task_type == 'clf' else mutual_info_regression
                
                # Convert to dense for mutual information calculation
                X_dense = X_filtered.toarray()
                
                # Use SelectPercentile instead of SelectKBest
                selector = SelectPercentile(
                    score_func=score_func,
                    percentile=50  # Select top 50% features by default
                )
                X_filtered = selector.fit_transform(X_dense, y)
                
                # Get selected feature indices
                selected_mask = selector.get_support()
                initial_features = initial_features[selected_mask]
                
                # Ensure minimum number of features
                if len(initial_features) < self.min_features:
                    # Switch to SelectKBest if needed
                    selector = SelectKBest(
                        score_func=score_func,
                        k=self.min_features
                    )
                    X_filtered = selector.fit_transform(X_dense, y)
                    selected_mask = selector.get_support()
                    initial_features = initial_features[selected_mask]
                
                # Store feature scores
                self.feature_stats[cluster_id] = {
                    'mutual_info_scores': dict(zip(initial_features, selector.scores_)),
                    'selected_features': initial_features.tolist(),
                    'n_selected_features': len(initial_features),
                    'selection_method': 'percentile' if len(initial_features) >= self.min_features else 'k_best'
                }
                
                self.logger.debug(f"Features after mutual information selection: {len(initial_features)}")
                self.logger.debug(f"Selection method: {self.feature_stats[cluster_id]['selection_method']}")
                
            except Exception as e:
                self.logger.error(f"Mutual information selection failed: {str(e)}")
                return None, []
        
        # QBSOFS feature selection
        if self.enable_qbsofs:
            try:
                # Calculate actual number of features to select
                n_features_to_select = min(
                    self.min_features,
                    X_filtered.shape[1]
                )
                
                selected_indices, fitness = self.qbsofs.select_features(
                    X_filtered, y, n_features_to_select
                )
                X_filtered = X_filtered[:, selected_indices]
                initial_features = initial_features[selected_indices]
                
                # Update feature statistics
                if cluster_id in self.feature_stats:
                    self.feature_stats[cluster_id].update({
                        'qbsofs_fitness': fitness,
                        'selected_features': initial_features.tolist()
                    })
                else:
                    self.feature_stats[cluster_id] = {
                        'qbsofs_fitness': fitness,
                        'selected_features': initial_features.tolist()
                    }
                
                self.logger.debug(f"Features after QBSOFS selection: {len(initial_features)}")
                
            except Exception as e:
                self.logger.error(f"QBSOFS failed: {str(e)}")
                return None, []
        
        # Convert back to sparse if needed
        if scipy.sparse.issparse(X) and not scipy.sparse.issparse(X_filtered):
            X_filtered = scipy.sparse.csr_matrix(X_filtered)
        
        return X_filtered, initial_features.tolist()
    
    def process_clusters(self,
                        balanced_clusters: Dict) -> Dict:
        """
        Process features for all clusters sequentially
        
        Args:
            balanced_clusters: Dictionary of cluster data
            
        Returns:
            Dictionary of processed cluster data
        """
        processed_clusters = {}
        
        for cluster_id, (X_cluster, y_cluster, _) in balanced_clusters.items():
            # Ensure y is 1-dimensional
            if y_cluster.ndim > 1:
                y_cluster = y_cluster.ravel()
            
            X_processed, selected_features = self._process_cluster_features(
                X_cluster, y_cluster, cluster_id
            )
            
            if X_processed is not None:
                processed_clusters[cluster_id] = (X_processed, y_cluster, selected_features)
                self.logger.info(f"Cluster {cluster_id}: Selected {len(selected_features)} features")
        
        return processed_clusters
    
    def _process_batch(self, batch_clusters):
        """Process a batch of clusters"""
        batch_results = {}
        for cluster_id, (X_cluster, y_cluster, _) in batch_clusters.items():
            if y_cluster.ndim > 1:
                y_cluster = y_cluster.ravel()
            
            try:
                X_processed, selected_features = self._process_cluster_features(
                    X_cluster, y_cluster, cluster_id
                )
                if X_processed is not None:
                    batch_results[cluster_id] = (
                        X_processed,
                        y_cluster,
                        selected_features
                    )
            except Exception as e:
                self.logger.error(f"Processing failed for cluster {cluster_id}: {str(e)}")
                continue
        
        return batch_results
    
    def process_clusters_parallel(self,
                                balanced_clusters: Dict,
                                n_jobs: int = -1) -> Dict:
        """
        Process features for all clusters in parallel
        
        Args:
            balanced_clusters: Dictionary of cluster data
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            
        Returns:
            Dictionary of processed cluster data
        """
        # Determine number of workers
        if n_jobs < 0:
            import multiprocessing
            n_jobs = min(multiprocessing.cpu_count(), len(balanced_clusters))
        else:
            n_jobs = min(n_jobs, len(balanced_clusters))
        
        self.logger.info(f"Processing {len(balanced_clusters)} clusters using {n_jobs} workers")
        
        # Process clusters in parallel
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        # Split clusters into batches
        cluster_items = list(balanced_clusters.items())
        batch_size = max(1, len(cluster_items) // n_jobs)
        batches = [
            dict(cluster_items[i:i + batch_size])
            for i in range(0, len(cluster_items), batch_size)
        ]
        
        processed_clusters = {}
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit batches
            future_to_batch = {
                executor.submit(self._process_batch, batch): batch 
                for batch in batches
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    processed_clusters.update(batch_results)
                    for cluster_id, (_, _, selected_features) in batch_results.items():
                        self.logger.info(f"Cluster {cluster_id}: Selected {len(selected_features)} features")
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {str(e)}")
                    continue
        
        return processed_clusters
    
    def get_feature_stats(self, cluster_id: int) -> Dict:
        """Get feature statistics for a specific cluster"""
        return self.feature_stats.get(cluster_id, {})