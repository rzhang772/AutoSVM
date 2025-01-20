import numpy as np
import scipy.sparse
from typing import Dict, List, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
import pandas as pd
from datetime import datetime

class FeatureIndependenceChecker:
    """Class for checking feature independence in clusters"""
    
    def __init__(self,
                 threshold: float = 0.5,  # Score threshold for feature selection
                 smooth_factor: float = 1.0,  # Smoothing factor (o)
                 n_jobs: int = -1,  # Number of parallel jobs
                 logger: logging.Logger = None):
        """
        Initialize FeatureIndependenceChecker
        
        Args:
            threshold: Score threshold for keeping features
            smooth_factor: Smoothing factor for calculation
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            logger: Logger instance from main
        """
        self.threshold = threshold
        self.smooth_factor = smooth_factor
        self.n_jobs = n_jobs
        self.logger = logger or logging.getLogger('feature_independence')
        self.cluster_stats = {}  # Store statistics for each cluster
        self.selected_features = {}  # Store selected features for each cluster
        self.feature_scores = {}  # Store independence scores for each cluster
    
    def _calculate_cluster_stats(self, cluster_data: Tuple) -> Dict:
        """
        Calculate non-zero and zero counts for a cluster
        
        Args:
            cluster_data: Tuple of (cluster_id, X, y)
            
        Returns:
            Dictionary containing cluster statistics
        """
        cluster_id, X, _ = cluster_data
        
        # Calculate non-zero counts for each feature separately
        nonzero_counts = []
        zero_counts = []
        
        for feat_idx in range(X.shape[1]):
            feature_col = X[:, feat_idx]
            if scipy.sparse.issparse(feature_col):
                feature_col = feature_col.toarray().flatten()
            
            nonzero = np.count_nonzero(feature_col)
            zero = X.shape[0] - nonzero
            
            nonzero_counts.append(nonzero)
            zero_counts.append(zero)
        
        stats = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'nonzero_counts': np.array(nonzero_counts),
            'zero_counts': np.array(zero_counts)
        }
        
        return {cluster_id: stats}
    
    def calculate_stats(self, clusters: Dict) -> None:
        """
        Calculate statistics for all clusters in parallel
        
        Args:
            clusters: Dictionary of cluster data
        """
        # Determine number of workers
        if self.n_jobs < 0:
            import multiprocessing
            self.n_jobs = multiprocessing.cpu_count()
        
        # Calculate statistics in parallel
        self.logger.info(f"Calculating cluster statistics using {self.n_jobs} workers...")
        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                try:
                    futures = {
                        executor.submit(
                            self._calculate_cluster_stats, 
                            (cid, data[0], data[1])
                        ): cid for cid, data in clusters.items()
                    }
                    
                    for future in as_completed(futures):
                        try:
                            stats = future.result()
                            self.cluster_stats.update(stats)
                        except Exception as e:
                            self.logger.error(f"Failed to calculate statistics: {str(e)}")
                finally:
                    executor.shutdown(wait=True)
                    import gc
                    gc.collect()
        except KeyboardInterrupt:
            print("Interrupted! Shutting down process pool...")
            executor.shutdown(wait=False)
    
    def fit_transform(self, clusters: Dict, dataset_name: str) -> Dict:
        """
        Check feature independence for all clusters
        
        Args:
            clusters: Dictionary of cluster data
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary of filtered cluster data
        """
        # First calculate statistics if not already done
        if not self.cluster_stats:
            self.calculate_stats(clusters)
        
        filtered_clusters = {}
        o = self.smooth_factor
        
        # Process each cluster using pre-calculated statistics
        for cluster_id, (X, y, _) in clusters.items():
            self.logger.debug(f"before independence check: {X.shape[1]} features")
            # Get target cluster stats
            target_stats = self.cluster_stats[cluster_id]
            target_nonzero = target_stats['nonzero_counts']
            target_zero = target_stats['zero_counts']
            
            # Combine other clusters' stats
            other_nonzero = np.zeros_like(target_nonzero)
            other_zero = np.zeros_like(target_zero)
            for other_id, other_stats in self.cluster_stats.items():
                if other_id != cluster_id:
                    other_nonzero += other_stats['nonzero_counts']
                    other_zero += other_stats['zero_counts']
            
            # Calculate independence scores
            selected_features_for_single_cluster = []
            scores = []
            
            for feat_idx in range(X.shape[1]):
                a = target_nonzero[feat_idx]
                b = other_nonzero[feat_idx]
                c = target_zero[feat_idx]
                d = other_zero[feat_idx]
                
                numerator = ((a + o) * (d + o) - (b + o) * (c + o)) ** 2
                denominator = ((a + b + o) * (a + c + o) * (c + b + o) * (c + d + o))
                score = numerator / denominator if denominator != 0 else 0
                
                scores.append(score)
                if score >= self.threshold:
                    selected_features_for_single_cluster.append(feat_idx)
            
            # Store scores
            self.feature_scores[cluster_id] = dict(enumerate(scores))
            # Store selected features for each cluster
            self.selected_features[cluster_id] = selected_features_for_single_cluster
            
            # Filter features
            if selected_features_for_single_cluster:
                X_filtered = X[:, selected_features_for_single_cluster]
                filtered_clusters[cluster_id] = (X_filtered, y, selected_features_for_single_cluster)
                self.logger.info(f"Cluster {cluster_id}: Selected {len(selected_features_for_single_cluster)} features")
                self.logger.debug(f"independence check: Selected features for cluster {cluster_id}: {selected_features_for_single_cluster}")
            else:
                filtered_clusters[cluster_id] = (X, y, list(range(X.shape[1])))
                self.logger.warning(f"Cluster {cluster_id}: No features passed independence check, keeping all features")
        
        # Save scores to file
        self.save_scores(dataset_name)
        
        return filtered_clusters
    
    def transform(self, clusters: Dict) -> Dict:
        """
        Transform clusters using selected features
        """
        clusters_filtered = {}
        for cluster_id, (X, y, _) in clusters.items():
            self.logger.debug(f"before independence check for cluster {cluster_id}: {X.shape[1]} features")
            if cluster_id in self.selected_features:
                if self.selected_features[cluster_id]: 
                    clusters_filtered[cluster_id] = (X[:, self.selected_features[cluster_id]], y, self.selected_features[cluster_id])
                else:
                    clusters_filtered[cluster_id] = (X, y, list(range(X.shape[1])))
            else:
                clusters_filtered[cluster_id] = (X, y, list(range(X.shape[1])))
            self.logger.debug(f"independence check: Selected features for cluster {cluster_id}: {clusters_filtered[cluster_id][2]}")  
        return clusters_filtered
    
    def save_scores(self, dataset_name: str) -> None:
        """
        Save feature independence scores to file
        
        Args:
            dataset_name: Name of the dataset
        """
        # Create output directory if not exists
        output_dir = "./output/independence_score"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert scores to DataFrame
        scores_data = []
        for cluster_id, scores in self.feature_scores.items():
            for feature_idx, score in scores.items():
                scores_data.append({
                    'cluster_id': cluster_id,
                    'feature_idx': feature_idx,
                    'score': score,
                    'selected': score >= self.threshold
                })
        
        df = pd.DataFrame(scores_data)
        
        # Save to CSV
        output_file = os.path.join(
            output_dir,
            f"{os.path.basename(dataset_name)}_independence_scores_{timestamp}.csv"
        )
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Independence scores saved to: {output_file}")