import numpy as np
import scipy.sparse
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, SelectPercentile
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict
from qbsofs import QBSOFS
from feature_constructor import FeatureConstructor
from itertools import compress


class FeatureProcessor:
    """Class for processing features in clustered subsets"""
    
    def __init__(self, 
                 task_type: str,           # 'clf' or 'reg'
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
            enable_mutual_info: Whether to enable mutual information feature selection
            enable_qbsofs: Whether to enable QBSOFS feature selection
            non_zero_threshold: Minimum ratio of non-zero values required
            min_features: Minimum number of features to keep
            qbsofs_params: Parameters for QBSOFS
            logger: Logger instance from main
        """
        self.task_type = task_type.lower()
        self.enable_mutual_info = enable_mutual_info
        self.enable_qbsofs = enable_qbsofs
        self.non_zero_threshold = non_zero_threshold
        self.min_features = min_features
        self.selected_features = {}
        
        # Initialize QBSOFS if enabled
        if self.enable_qbsofs:
            self.qbsofs = QBSOFS(**(qbsofs_params or {}))
        
        # Use provided logger or create new one
        self.logger = logger
        self.feature_stats = {}
        # self.selected_mask = {} # {cluster_id: [mutual_info_mask, qbsofs_mask]}
        
    
    def _process_cluster_features(self, 
                                X: scipy.sparse.csr_matrix,
                                y: np.ndarray,
                                cluster_id: int) -> Tuple[scipy.sparse.csr_matrix, List[int]]:
        """Process features for a single cluster"""
        X_filtered = X
        
        # Initial non-zero filtering
        # n_samples = X.shape[0]
        # non_zero_counts = np.array((X_filtered != 0).sum(axis=0)).flatten()
        # non_zero_ratios = non_zero_counts / n_samples
        
        # non_zero_mask = non_zero_ratios >= self.non_zero_threshold
        # if not any(non_zero_mask):
        #     self.logger.warning(f"No features passed non-zero threshold {self.non_zero_threshold}")
        #     return None, []
            
        # X_filtered = X_filtered[:, non_zero_mask]
        # self.non_zero_features = np.where(non_zero_mask)[0]
        
        # self.logger.debug(f"Features after non-zero filtering: {len(non_zero_features)}")
        
        # Mutual Information feature selection
        if self.enable_mutual_info:
            try:
                # Choose scoring function based on task type
                score_func = mutual_info_classif if self.task_type == 'clf' else mutual_info_regression
                
                # Convert to dense for mutual information calculation
                X_dense = X_filtered.toarray()
                
                # First try SelectPercentile
                percentile_selector = SelectPercentile(
                    score_func=score_func,
                    percentile=50  # Select top 50% features
                )
                X_percentile = percentile_selector.fit_transform(X_dense, y)
                self.logger.debug(f"X_percentile cluster{cluster_id}, new shape: {X_percentile.shape}")
                # Check if we have enough features
                if X_percentile.shape[1] >= self.min_features:
                    # Use SelectPercentile results
                    X_filtered = X_percentile
                    mutual_info_mask = percentile_selector.get_support()
                    selector_scores = percentile_selector.scores_
                    selection_method = 'percentile'
                else:
                    # Fall back to SelectKBest
                    kbest_selector = SelectKBest(
                        score_func=score_func,
                        k=self.min_features
                    )
                    X_filtered = kbest_selector.fit_transform(X_dense, y)
                    mutual_info_mask = kbest_selector.get_support()
                    selector_scores = kbest_selector.scores_
                    selection_method = 'k_best'
                mutual_select_features = np.where(mutual_info_mask)[0]
                
                # Store feature scores
                self.feature_stats[cluster_id] = {
                    'mutual_info_scores': dict(zip(mutual_select_features, selector_scores)),
                    'selected_features': mutual_select_features.tolist(),
                    'selection_method': selection_method
                }
                
                self.logger.debug(f"Features after mutual information selection: {len(mutual_select_features)}")
                self.logger.debug(f"Features after mutual information selection: {mutual_select_features}")
                self.logger.debug(f"Selection method used: {selection_method}")
                selected_features = mutual_select_features
                
            except Exception as e:
                self.logger.error(f"Mutual information selection failed: {str(e)}")
                return None, []
        
        # QBSOFS feature selection
        if self.enable_qbsofs:
            try:
                # TODO: Implement QBSOFS
                qbsofs_mask = None
                if qbsofs_mask is not None:
                    qbsofs_selected_features = np.where(qbsofs_mask)[0]
                    selected_features = list(compress(selected_features, qbsofs_mask))
            except Exception as e:
                self.logger.error(f"QBSOFS failed: {str(e)}")
                return None, []
            
        # Convert back to sparse if needed
        if scipy.sparse.issparse(X) and not scipy.sparse.issparse(X_filtered):
            X_filtered = scipy.sparse.csr_matrix(X_filtered)
        
        return X_filtered, selected_features 
    
    def fit_transform(self,
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
            
            X_processed, selected_features_for_single_cluster = self._process_cluster_features(
                X_cluster, y_cluster, cluster_id
            )
            self.logger.debug(f"Cluster {cluster_id}: Selected {len(selected_features_for_single_cluster)} features")
            
            # Store selected features for each cluster
            self.selected_features[cluster_id] = selected_features_for_single_cluster
            self.logger.debug(f"stored features for cluster {cluster_id}")
            
            if X_processed is not None:
                processed_clusters[cluster_id] = (X_processed, y_cluster, selected_features_for_single_cluster)
                self.logger.info(f"Cluster {cluster_id}: Selected {len(selected_features_for_single_cluster)} features")
        
        return processed_clusters
    
    def transform(self, clusters: Dict) -> Dict:
        """
        Transform existing features
        """
        processed_clusters = {} 
        self.logger.debug(f"selected features: {self.selected_features}")
        for cluster_id, (X_cluster, y_cluster, feature_indices) in clusters.items():
            self.logger.debug(f"before mutual and qbsofs for cluster {cluster_id}: {feature_indices}")
            if cluster_id in self.selected_features:
                new_feature_indices = self.selected_features[cluster_id]
            else:
                new_feature_indices = feature_indices
                self.logger.warning(f"No selected features for cluster {cluster_id}")

            X_processed = X_cluster[:, new_feature_indices]
            processed_clusters[cluster_id] = (X_processed, y_cluster, new_feature_indices)
            self.logger.debug(f"after mutual and qbsofs for cluster {cluster_id}: {new_feature_indices}")
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
    
    def fit_transform_parallel(self,
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
                    for cluster_id, (_, _, selected_features_for_single_cluster) in batch_results.items():
                        # Store selected features for each cluster
                        self.selected_features[cluster_id] = selected_features_for_single_cluster
                        self.logger.info(f"Cluster {cluster_id}: Selected {len(selected_features_for_single_cluster)} features")
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {str(e)}")
                    continue
        
        return processed_clusters
    
    def get_feature_stats(self, cluster_id: int) -> Dict:
        """Get feature statistics for a specific cluster"""
        return self.feature_stats.get(cluster_id, {})