import numpy as np
import pandas as pd
import scipy.sparse
import math

from typing import List, Tuple
from logger import Logger
from svmcluster import SVMCluster
from cluster_feature_selector import ClusterFeatureSelector

class CascadeCluster:
    '''
    Class for cascading clustering
    '''
    def __init__(self, 
                    cascade: bool = True,
                    entropy_selection: bool = False,
                    entropy_ratio: float = 0.3,
                    k: int = None,
                    K_MIN: int = 2,
                    K_MAX: int = 10,
                    method: str = 'kmeans',
                    algorithm: str = 'silhouette',
                    parallel_cluster: bool = False,
                    cluster_jobs: int = -1,
                 logger: Logger = None):
        '''
        Initialize CascadeCluster
        
        Args:
            logger: Logger instance
        '''
        self.logger = logger or Logger.get_logger('cascade_cluster')
        self.selector = None
        self.clusterer0 = None
        self.clusterers = []
        self.over_labels = []
        self.k_level1 = {}

        self.cascade = cascade
        self.entropy_selection = entropy_selection
        self.entropy_ratio = entropy_ratio
        self.k = k
        self.K_MIN = K_MIN
        self.K_MAX = K_MAX
        self.method = method
        self.algorithm = algorithm
        self.parallel_cluster = parallel_cluster
        self.cluster_jobs = cluster_jobs

        
    
    def fit_predict(self, X: scipy.sparse.csr_matrix, max_cluster_ratio:float = 0.1):
        '''
        Fit the cascade clusterer on data and transform it
        
        Args:
            X: Input feature matrix
            y: Input target vector
            n_clusters: Number of clusters
            clusterer: Clusterer instance
        '''
        try:
            
            self.logger.info(f"Fitting cascade clusterer..., level:0")
            if self.entropy_selection:
                self.selector = ClusterFeatureSelector(select_ratio=self.entropy_ratio, logger=self.logger)
                X_train_normalized_selected, _ = self.selector.fit_transform(X, select_strategy='bottom')  
            else:
                X_train_normalized_selected = X

            # Initialize clusterer with main logger
            self.clusterer0 = SVMCluster(
                logger=self.logger  # Pass the main logger
            )
            best_k, results, labels, _ = self.clusterer0.fit_predict(
                X_train_normalized_selected,
                k=self.k,
                k_range=range(self.K_MIN, self.K_MAX + 1) if self.k is None else None,
                method=self.method,
                algorithm=self.algorithm,
                parallel=self.parallel_cluster,  # Use clustering-specific parallel parameter
                n_jobs=self.cluster_jobs       # Use clustering-specific jobs parameter
            )

            # detect over-sized clusters
            over_size = {}
            unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
            self.logger.debug("\nCluster size distribution after level 0:")
            for label, size in zip(unique_labels, cluster_sizes):
                if size > int(max_cluster_ratio*X.shape[0]):
                    self.over_labels.append(label)
                    over_size[label] = size
                self.logger.debug(f"Cluster {label}: {size} samples ({size/len(labels)*100:.2f}%)")

            self.logger.info(f"Over-sized clusters: {over_size}\n Starting level 1...")
            df = pd.DataFrame({'level 0': labels})
            if len(self.over_labels) == 0 or not self.cascade:
                self.logger.info("No over-sized clusters detected. Returning level 0 labels as final labels.")
                df['level 1'] = df['level 0']
                df['final_label'] = df['level 0']
                return labels, df
            else:
                # calculate the k for each over-sized cluster
                for over_label in self.over_labels:
                    self.k_level1[over_label] = math.ceil(over_size[over_label]/(X.shape[0]*max_cluster_ratio))
                self.logger.debug(f"K for level 1: {self.k_level1}")
                
                sub_labels = {}
                for over_label in self.over_labels:
                    # filter the over-sized cluster
                    X_filtered = X[labels == over_label]
                    # Initialize clusterer with main logger
                    clusterer1 = SVMCluster(
                        logger=self.logger  # Pass the main logger
                    )
                    best_k, results, sub_label, _ = clusterer1.fit_predict(
                        X_filtered,
                        k=self.k_level1[over_label],
                        method=self.method,
                    )
                    sub_labels[over_label] = sub_label
                    self.clusterers.append(clusterer1)
                # merge the labels
                # Step 2: Initialize the `level 1` column with NaN
                df['level 1'] = np.nan

                # Step 3: Fill `level 1` for oversized clusters
                for cluster, sub_label in sub_labels.items():
                    # Locate rows belonging to the oversized cluster
                    indices = np.where(labels == cluster)[0]
                    # Assign sub-labels to the corresponding rows in 'level 1'
                    df.loc[indices, 'level 1'] = sub_label

                # Step 4: Fill `level 1` for non-oversized clusters with their `level 0` value
                df['level 1'] = df['level 1'].fillna(df['level 0']).astype(int)

                # Combine `level 0` and `level 1` into tuples
                df['combined_label'] = list(zip(df['level 0'], df['level 1']))

                # Map unique combinations to incrementally increasing integers
                unique_labels = {label: i for i, label in enumerate(sorted(df['combined_label'].unique()))}
                df['final_label'] = df['combined_label'].map(unique_labels)

                # Drop the intermediate column if no longer needed
                df = df.drop(columns=['combined_label'])

            return df['final_label'].values,df
        
        except Exception as e:
            self.logger.error(f"Error in fitting cascade clusterer: {e}")
            raise e
    
    def predict(self, X: scipy.sparse.csr_matrix):

        if self.entropy_selection:
            X_selected = self.selector.transform(X)
        else:
            X_selected = X
        labels_level0 = self.clusterer0.predict(X_selected)
        df = pd.DataFrame({'level 0': labels_level0})

        if(len(self.over_labels) == 0 or not self.cascade):
            self.logger.info("No over-sized clusters detected or cascede deactivated. Returning level 0 labels as final labels.")
            df['level 1'] = df['level 0']
            df['final_label'] = df['level 0']
            return df['final_label'].values, df
        else:
            sub_labels = {}
            for over_label, clusterer in zip(self.over_labels, self.clusterers):
                X_filtered = X[labels_level0 == over_label]
                sub_labels[over_label] = clusterer.predict(X_filtered)
            # merge the labels
            
            # Step 2: Initialize the `level 1` column with NaN
            df['level 1'] = np.nan

            # Step 3: Fill `level 1` for oversized clusters
            for cluster, sub_label in sub_labels.items():
                # Locate rows belonging to the oversized cluster
                indices = np.where(labels_level0 == cluster)[0]
                # Assign sub-labels to the corresponding rows in 'level 1'
                df.loc[indices, 'level 1'] = sub_label

            # Step 4: Fill `level 1` for non-oversized clusters with their `level 0` value
            df['level 1'] = df['level 1'].fillna(df['level 0']).astype(int)

            # Combine `level 0` and `level 1` into tuples
            df['combined_label'] = list(zip(df['level 0'], df['level 1']))

            # Map unique combinations to incrementally increasing integers
            unique_labels = {label: i for i, label in enumerate(sorted(df['combined_label'].unique()))}
            df['final_label'] = df['combined_label'].map(unique_labels)

            # Drop the intermediate column if no longer needed
            df = df.drop(columns=['combined_label'])

        return df['final_label'].values, df
