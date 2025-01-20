import numpy as np
import pandas as pd
import scipy.sparse
import math
import time
import os

from typing import List, Tuple
from logger import Logger
from svmcluster import SVMCluster
from cluster_feature_selector import ClusterFeatureSelector

MAX_SIZE = 10000

class CascadeCluster:
    '''
    Class for cascading clustering
    '''
    def __init__(self, 
                    train: str,
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
        self.train = train
        self.logger = logger or Logger.get_logger('cascade_cluster')
        self.selector = None
        self.clusterer0 = None
        self.best_k = None
        self.clusterers = []
        self.over_labels = []
        self.k_level1 = {}
        self.k_level1_out = {}
        self.label_mapping = {}

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
                logger=self.logger,  # Pass the main logger
                sample_ratio=0.1
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
            self.best_k = best_k
            unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
            # df_labels['cluster_size'] = cluster_sizes
            self.logger.info(f"cluster number:{len(unique_labels)}")

            # self.logger.debug(f"\nCluster size distribution:")
            # for label, size in zip(unique_labels, cluster_sizes):
            #     self.logger.debug(f"Cluster {label}: {size} samples ({size/len(labels)*100:.2f}%)")
            # exit(0)

            # Save scores for k results
            if self.algorithm == 'kmeans' and self.k is None:
                save_start = time.time()
                self.clusterer0.save_results(
                    os.path.basename(self.train),
                    best_k,
                    results,
                    self.method
                )
                save_time = time.time() - save_start
                self.logger.info(f"Results for k scores saved in {save_time:.2f} seconds")
            

            # detect over-sized clusters
            over_size = {}
            unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
            self.logger.info(f"Clusters after level 0: {len(unique_labels)}")
            for label, size in zip(unique_labels, cluster_sizes):
                self.logger.debug(f"Cluster {label}: {size} samples ({size/len(labels)*100:.2f}%)")
            if(len(unique_labels) != best_k):
                self.logger.error(f"Best k: {best_k}, Unique labels: {len(unique_labels)}")
                raise ValueError("Number of clusters not equal to best k")
            self.logger.debug("\nCluster size distribution after level 0:")
            for label, size in zip(unique_labels, cluster_sizes):
                # if size > int(max_cluster_ratio*X.shape[0]):
                if size > MAX_SIZE:
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
                    # self.k_level1[over_label] = math.ceil(over_size[over_label]/(X.shape[0]*max_cluster_ratio))
                    self.k_level1[over_label] = math.ceil(over_size[over_label]/MAX_SIZE)
                self.logger.debug(f"K for level 1: {self.k_level1}")
                
                sub_labels = {}
                for over_label in self.over_labels:
                    # filter the over-sized cluster
                    X_filtered = X[labels == over_label]
                    # Initialize clusterer with main logger
                    clusterer1 = SVMCluster(
                        logger=self.logger  # Pass the main logger
                    )
                    best_k1, results1, sub_label, _ = clusterer1.fit_predict(
                        X_filtered,
                        k=self.k_level1[over_label],
                        method=self.method,
                    )
                    self.k_level1_out[over_label] = len(np.unique(sub_label))
                    self.logger.info(f"k for {over_label} in level 1: {best_k1}, labels:{np.unique(sub_label)}")
                    sub_labels[over_label] = sub_label
                    self.clusterers.append(clusterer1)

                    if best_k1 != self.k_level1[over_label]:
                        self.logger.warning(f"Best k: {best_k1}, Expected k: {self.k_level1[over_label]}")
                    #     raise ValueError("Number of clusters not equal to best k")
                
                
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
                df['level 1'] = df['level 1'].fillna(-1).astype(int)

                current_label = 0
                # Step 3: Loop through each first-layer cluster
                for level0_label in range(best_k):
                    if level0_label in self.k_level1_out:
                        # If this cluster is oversized and has second-layer clustering
                        level1_count = self.k_level1_out[level0_label]
                        for level1_label in range(level1_count):
                            # Map (level 0, level 1) to the current global label
                            self.label_mapping[(level0_label, level1_label)] = current_label
                            current_label += 1
                    else:
                        # If this cluster is not oversized
                        # Map (level 0, -1) to the current global label
                        self.label_mapping[(level0_label, -1)] = current_label
                        current_label += 1
                self.logger.debug(f"Label mapping: {self.label_mapping}")

                df['final_label'] = df.apply(
                        lambda row: self.label_mapping.get((row['level 0'], row['level 1']), -1), axis=1
                    )
                self.logger.info(f"Final labels: {len(np.unique(df['final_label'].values))}")
                # if sum(self.k_level1.values()) - len(self.over_labels) + best_k != len(np.unique(df['final_label'].values)):
                #     self.logger.error(f"Number of unique labels: {len(np.unique(df['final_label'].values))}, Expected: {sum(self.k_level1.values()) - len(self.over_labels) + best_k}")
                #     raise ValueError("Number of unique labels not equal to expected")

            return df['final_label'].values,df
        
        except Exception as e:
            self.logger.error(f"Error in fitting cascade clusterer: {e}")
            raise e
    
    def predict(self, X: scipy.sparse.csr_matrix):

        if self.entropy_selection:
            X_selected = self.selector.transform(X)
        else:
            X_selected = X
        self.logger.debug(f"selected shape: {X_selected.shape}")
        labels_level0 = self.clusterer0.predict(X_selected)
        self.logger.debug(f"len: {len(labels_level0)}, level 0 labels: {np.unique(labels_level0)}") 

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
                self.logger.debug(f"{over_label}, X_filtered: {X_filtered.shape}")
                if X_filtered.shape[0] == 0:
                    self.logger.warning(f"Empty cluster: {over_label}")
                    sub_labels[over_label] = np.array([])
                    continue
                sub_labels[over_label] = clusterer.predict(X_filtered)
                self.logger.debug(f"sub_labels: {np.unique(sub_labels[over_label])}")
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
            df['level 1'] = df['level 1'].fillna(-1).astype(int)

            df['final_label'] = df.apply(
                    lambda row: self.label_mapping.get((row['level 0'], row['level 1']), -1), axis=1
                )
            self.logger.info(f"Final testset labels: {len(np.unique(df['final_label'].values))}")

        return df['final_label'].values, df
