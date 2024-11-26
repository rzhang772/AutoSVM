import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from dataset import Dataset
from logger import Logger
import scipy.sparse
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

class ClusterMethod(Enum):
    KMEANS = "kmeans"
    RANDOM = "random"
    FIFO = "fifo"

class BaseCustomClusterer:
    """
    Base class for clustering algorithms
    
    Implements basic fit/predict interface. Subclasses need to implement specific clustering logic.
    """
    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
        
    def fit_predict(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        """Fit and predict in one step"""
        raise NotImplementedError
        
    def fit(self, X: scipy.sparse.csr_matrix) -> 'BaseCustomClusterer':
        """Fit the clusterer"""
        self.labels_ = self.fit_predict(X)
        return self
        
    def predict(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        """Predict using fitted clusterer"""
        raise NotImplementedError

class RandomClusterer(BaseCustomClusterer):
    """Random average clustering"""
    def fit_predict(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        base_size = n_samples // self.n_clusters
        remainder = n_samples % self.n_clusters
        
        cluster_sizes = np.full(self.n_clusters, base_size)
        cluster_sizes[:remainder] += 1
        
        perm = np.random.permutation(n_samples)
        
        start = 0
        for i in range(self.n_clusters):
            end = start + cluster_sizes[i]
            labels[perm[start:end]] = i
            start = end
            
        self.labels_ = labels
        return labels
        
    def predict(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        """Assign new data points to clusters randomly"""
        n_samples = X.shape[0]
        return np.random.randint(0, self.n_clusters, size=n_samples)

class FIFOClusterer(BaseCustomClusterer):
    """First-come-first-serve clustering"""
    def fit_predict(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        n_samples = X.shape[0]
        self.labels_ = np.arange(n_samples) % self.n_clusters
        return self.labels_
        
    def predict(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        """Assign new data points using modulo operation"""
        n_samples = X.shape[0]
        return np.arange(n_samples) % self.n_clusters

class SVMCluster:
    """Class for clustering analysis of SVM datasets"""
    
    def __init__(self, 
                 output_dir: str = "./output/cluster",
                 random_state: int = 42,
                 sample_ratio: float = 0.1,
                 enable_sampling: bool = True,
                 batch_size: int = 1024,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize SVMCluster
        
        Args:
            output_dir: Directory for saving results
            random_state: Random seed for reproducibility
            sample_ratio: Ratio of data to use for large datasets
            enable_sampling: Whether to enable sampling for large datasets
            batch_size: Batch size for MiniBatchKMeans
            logger: Logger instance from main
        """
        self.output_dir = output_dir
        self.random_state = random_state
        self.sample_ratio = sample_ratio
        self.enable_sampling = enable_sampling
        self.batch_size = batch_size
        self.model = None
        
        # Set random seed
        np.random.seed(random_state)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Use provided logger or create new one
        self.logger = logger or logging.getLogger('cluster')
        
    def preprocess_data(self, X: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
        """Preprocess data for clustering"""
        # Sample large datasets if sampling is enabled
        if self.enable_sampling and X.shape[0] > 100000:
            sample_size = int(X.shape[0] * self.sample_ratio)
            self.logger.info(f"Sampling {sample_size} instances from {X.shape[0]} total instances")
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X = X[indices]
        else:
            if X.shape[0] > 100000:
                self.logger.warning(
                    f"Processing large dataset with {X.shape[0]} instances without sampling. "
                    "This may take a long time and consume significant memory."
                )
        
        return X
    
    def find_optimal_k(self, 
                      X: scipy.sparse.csr_matrix, 
                      k_range: range,
                      method: str = "silhouette",
                      parallel: bool = False,
                      n_jobs: int = -1) -> Tuple[int, Dict]:
        """
        Find optimal number of clusters
        
        Args:
            X: Input data (sparse matrix)
            k_range: Range of k values to try
            method: Method to use ('silhouette' or 'gap')
            parallel: Whether to use parallel processing
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            
        Returns:
            Tuple of (optimal k, method results)
        """
        if method.lower() == "silhouette":
            if parallel:
                scores = self._calculate_silhouette_parallel(X, k_range, n_jobs)
            else:
                scores = self._calculate_silhouette(X, k_range)
            best_k = k_range[np.argmax(scores)]
            results = {"k": list(k_range), "scores": scores}
        else:  # gap statistic
            if parallel:
                gap_values, sk_values, best_k = self._calculate_gap_parallel(X, k_range, n_jobs)
            else:
                gap_values, sk_values, best_k = self._calculate_gap(X, k_range)
            results = {
                "k": list(k_range),
                "gap_values": gap_values,
                "sk_values": sk_values
            }
        
        return best_k, results
    
    def _calculate_silhouette(self, 
                            X: scipy.sparse.csr_matrix, 
                            k_range: range) -> List[float]:
        """Calculate silhouette scores"""
        scores = []
        best_score = -1
        best_k = k_range[0]
        
        for k in k_range:
            # self.logger.info(f"Computing silhouette score for k={k}")
            kmeans = MiniBatchKMeans(
                n_clusters=k, 
                random_state=self.random_state,
                batch_size=self.batch_size
            )
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels, metric="euclidean", sample_size=10000)
            scores.append(score)
            
            self.logger.debug(f"k={k}: silhouette score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
                
        self.logger.info(f"\nBest silhouette score: {best_score:.4f} (k={best_k})")
        return scores
    
    def _calculate_gap(self, 
                      X: scipy.sparse.csr_matrix, 
                      k_range: range, 
                      n_refs: int = 10) -> Tuple[List[float], List[float], int]:
        """Calculate GAP statistics"""
        # Initial memory check
        try:
            required_memory = self._estimate_gap_memory(X, k_range, n_refs)
            initial_available = self._get_available_memory()
            
            self.logger.debug(f"Estimated memory requirement: {required_memory / (1024**3):.2f} GB")
            self.logger.debug(f"Available memory: {initial_available / (1024**3):.2f} GB")
            
            if required_memory > initial_available * 0.9:
                error_msg = (
                    f"Insufficient memory for GAP calculation. "
                    f"Required: {required_memory / (1024**3):.2f} GB, "
                    f"Available: {initial_available / (1024**3):.2f} GB. "
                    "Consider using sampling or reducing dataset size."
                )
                self.logger.error(error_msg)
                raise MemoryError(error_msg)
            
        except ImportError:
            self.logger.error("psutil not installed. Cannot proceed with GAP calculation without memory verification.")
            raise
        except Exception as e:
            self.logger.error(f"Memory check failed: {str(e)}")
            raise
        
        self.logger.info("Initial memory check passed. Starting GAP calculation...")
        
        gap_values = []
        sk_values = []
        best_gap = -float('inf')
        best_k = k_range[0]
        
        try:
            # Get dense reference data bounds
            X_sample = X[np.random.choice(X.shape[0], min(10000, X.shape[0]), replace=False)]
            X_sample = X_sample.toarray()
            mins = X_sample.min(axis=0)
            maxs = X_sample.max(axis=0)
            
            # Monitor memory during calculation
            for k in k_range:
                # self.logger.info(f"Computing GAP statistic for k={k}")
                
                # Check current memory usage
                current_memory = self._get_available_memory()
                memory_used = initial_available - current_memory
                memory_usage_ratio = memory_used / required_memory
                
                if memory_usage_ratio > 1.1:  # Memory usage exceeds estimate by 10%
                    error_msg = (
                        f"Memory usage exceeded estimate by more than 10%. "
                        f"Used: {memory_used / (1024**3):.2f} GB, "
                        f"Estimated: {required_memory / (1024**3):.2f} GB"
                    )
                    self.logger.error(error_msg)
                    raise MemoryError(error_msg)
                
                # Compute dispersion for actual data
                kmeans = MiniBatchKMeans(
                    n_clusters=k, 
                    random_state=self.random_state,
                    batch_size=self.batch_size
                )
                labels = kmeans.fit_predict(X)
                disp_actual = self._compute_dispersion(X, labels, kmeans.cluster_centers_)
                
                # Compute dispersions for reference data
                disp_refs = []
                for _ in range(n_refs):
                    # Check memory again before generating reference data
                    current_memory = self._get_available_memory()
                    memory_used = initial_available - current_memory
                    memory_usage_ratio = memory_used / required_memory
                    
                    if memory_usage_ratio > 1.1:
                        error_msg = (
                            f"Memory usage exceeded estimate during reference data generation. "
                            f"Used: {memory_used / (1024**3):.2f} GB, "
                            f"Estimated: {required_memory / (1024**3):.2f} GB"
                        )
                        self.logger.error(error_msg)
                        raise MemoryError(error_msg)
                    
                    ref_data = scipy.sparse.random(
                        X.shape[0], X.shape[1],
                        density=X.nnz/(X.shape[0]*X.shape[1]),
                        random_state=self.random_state
                    ).tocsr()
                    ref_labels = kmeans.fit_predict(ref_data)
                    disp_refs.append(
                        self._compute_dispersion(ref_data, ref_labels, kmeans.cluster_centers_)
                    )
                
                # Calculate GAP and SK values
                gap = np.mean(np.log(disp_refs)) - np.log(disp_actual)
                sk = np.std(np.log(disp_refs)) * np.sqrt(1 + 1/n_refs)
                
                gap_values.append(gap)
                sk_values.append(sk)
                
                self.logger.debug(f"k={k}: gap = {gap:.4f}, sk = {sk:.4f}")
                
                if gap > best_gap:
                    best_gap = gap
                    best_k = k
                
                self.logger.debug(f"Memory usage: {memory_used / (1024**3):.2f} GB "
                               f"({memory_usage_ratio * 100:.1f}% of estimate)")
            
            self.logger.info(f"\nBest GAP value: {best_gap:.4f} (k={best_k})")
            
            # Find optimal k using GAP criterion
            optimal_k = k_range[0]
            for i in range(len(gap_values) - 1):
                if gap_values[i] >= gap_values[i + 1] - sk_values[i + 1]:
                    optimal_k = k_range[i]
                    break
            
            self.logger.info(f"Optimal k by GAP criterion: {optimal_k}")
            return gap_values, sk_values, optimal_k
            
        except MemoryError as me:
            self.logger.error("GAP calculation terminated due to memory constraints")
            raise
        except Exception as e:
            self.logger.error(f"Error during GAP calculation: {str(e)}")
            raise
    
    def _compute_dispersion(self, 
                          X: scipy.sparse.csr_matrix, 
                          labels: np.ndarray,
                          centers: np.ndarray) -> float:
        """
        Compute cluster dispersion for sparse data
        
        Args:
            X: Input data (sparse matrix)
            labels: Cluster labels
            centers: Cluster centers
        """
        dispersion = 0
        for k in np.unique(labels):
            cluster_mask = labels == k
            if np.sum(cluster_mask) > 1:
                # Calculate distances to center efficiently for sparse data
                center = centers[k]
                cluster_points = X[cluster_mask]
                
                # Compute squared distances efficiently
                distances = cluster_points.multiply(cluster_points).sum(axis=1) \
                          + np.sum(center**2) \
                          - 2 * cluster_points.dot(center)
                
                dispersion += np.sum(distances)
        return dispersion
    
    def save_results(self, 
                    dataset_name: str, 
                    best_k: int, 
                    results: Dict,
                    method: str) -> None:
        """
        Save clustering results to file and log to main logger
        
        Args:
            dataset_name: Name of the dataset
            best_k: Best number of clusters found
            results: Dictionary containing results
            method: Method used ('silhouette' or 'gap')
        """
        # Get logger from main
        self.logger = logging.getLogger('main')
        
        # Save detailed results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(
            self.output_dir, 
            f"{dataset_name}_{method}_{timestamp}.csv"
        )
        pd.DataFrame(results).to_csv(results_file, index=False)
        
        # Log results to main logger
        # self.logger.info(f"Dataset: {dataset_name}")
        # self.logger.info(f"Best k: {best_k}")
        
        # if method == "silhouette":
        #     self.logger.info("Silhouette scores:")
        #     for k, score in zip(results['k'], results['scores']):
        #         self.logger.info(f"  k={k}: {score:.4f}")
        # else:  # gap statistic
        #     self.logger.info("GAP statistics:")
        #     for k, gap, sk in zip(results['k'], results['gap_values'], results['sk_values']):
        #         self.logger.info(f"  k={k}: gap={gap:.4f}, sk={sk:.4f}")
                
        self.logger.info(f"Detailed scores results saved to: {results_file}")
    
    def fit_predict(self,
                X: scipy.sparse.csr_matrix,
                k: Optional[int] = None,
                k_range: Optional[range] = None,
                method: str = "silhouette",
                algorithm: str = "kmeans",
                parallel: bool = False,
                n_jobs: int = -1) -> Tuple[int, Dict, np.ndarray, Any]:
        """
        Perform clustering analysis
        
        Args:
            X: Input data (sparse matrix)
            k: Fixed number of clusters (optional)
            k_range: Range of k values to try (optional)
            method: Method for finding optimal k ('silhouette' or 'gap')
            algorithm: Clustering algorithm ('kmeans', 'random', 'fifo')
            parallel: Whether to use parallel processing
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            
        Returns:
            Tuple of (best k, method results, cluster labels, clustering model)
        """
        # Preprocess data
        X = self.preprocess_data(X)
        
        # Determine k value
        if k is None:
            if k_range is None:
                raise ValueError("Either k or k_range must be provided")
            best_k, results = self.find_optimal_k(X, k_range, method, parallel, n_jobs)
        else:
            best_k = k
            results = {}
        
        # Perform clustering
        if algorithm == "kmeans":
            self.model = MiniBatchKMeans(
                n_clusters=best_k,
                random_state=self.random_state,
                batch_size=self.batch_size
            )
            labels = self.model.fit_predict(X)
        elif algorithm == "random":
            self.model = RandomClusterer(n_clusters=best_k, random_state=self.random_state)
            labels = self.model.fit_predict(X)
        else:  # fifo
            self.model = FIFOClusterer(n_clusters=best_k, random_state=self.random_state)
            labels = self.model.fit_predict(X)
        
        return best_k, results, labels, self.model
    
    def _calculate_gap_for_k(self,
                           X: scipy.sparse.csr_matrix,
                           k: int,
                           n_refs: int = 10) -> Tuple[float, float, float]:
        """
        Calculate GAP statistic for a single k value
        
        Returns:
            Tuple[float, float, float]: (gap value, sk value, actual data dispersion)
        """
        # Get dense reference data bounds
        X_sample = X[np.random.choice(X.shape[0], min(10000, X.shape[0]), replace=False)]
        X_sample = X_sample.toarray()
        mins = X_sample.min(axis=0)
        maxs = X_sample.max(axis=0)
        
        # Compute dispersion for actual data
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=self.random_state,
            batch_size=self.batch_size
        )
        labels = kmeans.fit_predict(X)
        disp_actual = self._compute_dispersion(X, labels, kmeans.cluster_centers_)
        
        # Compute dispersions for reference data
        disp_refs = []
        for _ in range(n_refs):
            ref_data = scipy.sparse.random(
                X.shape[0], X.shape[1],
                density=X.nnz/(X.shape[0]*X.shape[1]),
                random_state=self.random_state
            ).tocsr()
            ref_labels = kmeans.fit_predict(ref_data)
            disp_refs.append(
                self._compute_dispersion(ref_data, ref_labels, kmeans.cluster_centers_)
            )
        
        # Calculate GAP and SK values
        gap = np.mean(np.log(disp_refs)) - np.log(disp_actual)
        sk = np.std(np.log(disp_refs)) * np.sqrt(1 + 1/n_refs)
        
        return gap, sk, disp_actual 
    
    def _estimate_gap_memory(self, X: scipy.sparse.csr_matrix, k_range: range, n_refs: int) -> int:
        """
        Estimate memory requirements for GAP calculation
        
        Args:
            X: Input data matrix
            k_range: Range of k values to try
            n_refs: Number of reference distributions
            
        Returns:
            Estimated memory requirement in bytes
        """
        # Memory for reference datasets
        ref_data_size = X.data.nbytes * n_refs
        
        # Memory for cluster centers (dense)
        max_k = max(k_range)
        centers_size = X.shape[1] * max_k * 8  # 8 bytes per float64
        
        # Memory for labels and intermediate calculations
        labels_size = X.shape[0] * 4  # 4 bytes per int32
        intermediate_size = X.shape[0] * 8  # 8 bytes per float64 for distances
        
        # Memory for results
        results_size = len(k_range) * 8 * 2  # gap and sk values
        
        # Total memory with safety factor
        total_size = (ref_data_size + centers_size + labels_size + 
                     intermediate_size + results_size) * 1.5  # 50% safety margin
        
        return int(total_size)
    
    def _get_available_memory(self) -> int:
        """
        Get available system memory
        
        Returns:
            Available memory in bytes
        """
        try:
            import psutil
            vm = psutil.virtual_memory()
            return vm.available
        except ImportError:
            self.logger.warning("psutil not installed. Cannot check available memory.")
            return float('inf')  # Assume infinite memory if can't check
    
    def _process_k_batch(self, k_batch, X):
        """Process a batch of k values"""
        batch_results = []
        for k in k_batch:
            kmeans = MiniBatchKMeans(
                n_clusters=k, 
                random_state=self.random_state,
                batch_size=self.batch_size
            )
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels, metric="euclidean", sample_size=10000)
            batch_results.append((k, score))
        return batch_results

    def _calculate_silhouette_parallel(self, 
                                     X: scipy.sparse.csr_matrix, 
                                     k_range: range,
                                     n_jobs: int) -> List[float]:
        """Calculate silhouette scores in parallel"""
        if n_jobs < 0:
            import multiprocessing
            n_jobs = min(multiprocessing.cpu_count(), len(k_range))
        else:
            n_jobs = min(n_jobs, len(k_range))
        
        scores = [0.0] * len(k_range)
        best_score = -1
        best_k = k_range[0]
        
        self.logger.info(f"Starting parallel silhouette calculation with {n_jobs} workers")
        
        # Split k values into batches
        batch_size = max(1, len(k_range) // n_jobs)
        k_batches = [
            list(k_range[i:i + batch_size])
            for i in range(0, len(k_range), batch_size)
        ]
        
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_batch = {
                executor.submit(self._process_k_batch, batch, X): batch 
                for batch in k_batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    for k, score in batch_results:
                        idx = k_range.index(k)
                        scores[idx] = score
                        
                        # self.logger.info(f"k={k}: silhouette score = {score:.4f}")
                        
                        if score > best_score:
                            best_score = score
                            best_k = k
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {str(e)}")
                    continue
        
        self.logger.info(f"\nBest silhouette score: {best_score:.4f} (k={best_k})")
        return scores
    
    
    def predict(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        """
        Predict new data using fitted clustering model
        
        Args:
            X: Input feature matrix
            
        Returns:
            Cluster labels for input data
        """
        if self.model is None:
            raise ValueError("Clustering model not fitted. Call fit_predict first.")
        labels = self.model.predict(X)
        for cluster_id in np.unique(labels):
            self.logger.debug(f"Cluster {cluster_id}: {len(labels[labels == cluster_id])} samples") 
        return labels