import numpy as np
import scipy.sparse
from typing import Dict, List
import logging
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer


class FeatureConstructor:
    """Class for constructing new features from existing ones"""
    
    def __init__(self, n_bins: int = 5, discretize_ratio: float = 0.3, logger: logging.Logger = None):
        """
        Initialize FeatureConstructor
        
        Args:
            n_bins: Number of bins for discretization
            discretize_ratio: Ratio of features to discretize (top features by range)
        """
        self.n_bins = n_bins
        self.discretize_ratio = discretize_ratio
        # standardization scaler
        self.scaler = {}

        # discretize
        self.discretizer = {}
        self.top_features = {}

        # feature interactions
        self.min_features_for_interaction = 5
        self.feature_interaction_matrix = {}

        self.logger = logger

    def feature_interaction(self, X_dense:np.ndarray, cluster_id: int) -> List[scipy.sparse.csr_matrix]:
        new_features = []
        for i in range(min(self.min_features_for_interaction, X_dense.shape[1])):
            for j in range(i+1, min(self.min_features_for_interaction+1, X_dense.shape[1])):
                # Multiplication
                mult = X_dense[:, i] * X_dense[:, j]
                new_features.append(scipy.sparse.csr_matrix(mult.reshape(-1, 1)))
                
                # Division (safe)
                with np.errstate(divide='ignore', invalid='ignore'):
                    div = np.where(X_dense[:, j] != 0, 
                                X_dense[:, i] / X_dense[:, j],
                                0)
                div = np.nan_to_num(div)
                new_features.append(scipy.sparse.csr_matrix(div.reshape(-1, 1)))
        self.logger.debug(f"Feature interaction, new feature_num: {len(new_features)}") 
        return new_features
        
    def fit_transform(self, clusters: Dict) -> Dict:
        """
        Construct new features from input matrix
        
        Args:
            clusters: Dictionary of cluster data {cluster_id: (X, y, feature_indices)}
            
        Returns:
            Dictionary of cluster data with new features {cluster_id: (augmented feature matrix, y, feature_indices)}
        """
        
        try:
            for cluster_id, (X, y, feature_indices) in clusters.items():
                new_features = []
                max_feature_index = max(feature_indices) if feature_indices else 0
                new_feature_indices = feature_indices.copy()
                X_dense = X.toarray()
                self.logger.debug(f"original X_dense cluster{cluster_id}, shape: {X_dense.shape}, feature_indices: {feature_indices}")
                self.logger.debug(f"before feature construction for cluster {cluster_id}: {feature_indices}")
                new_feature_num = 0
                # 1. Standardization of original features
                scaler_for_single_cluster = StandardScaler()
                X_scaled = scaler_for_single_cluster.fit_transform(X_dense)
                new_features.append(scipy.sparse.csr_matrix(X_scaled))
                self.scaler[cluster_id] = scaler_for_single_cluster
                self.logger.debug(f"Scaler cluster{cluster_id}, new feature: {X_scaled.shape}")
                new_feature_num += X_scaled.shape[1]
                # self.logger.debug(f"new_features cluster{cluster_id}, new shape: {len(new_features)}")
            
                # 2. Select features for discretization based on range
                feature_ranges = np.ptp(X_dense, axis=0)  # Calculate range for each feature
                n_features_to_discretize = int(np.ceil(X.shape[1] * self.discretize_ratio))
                top_features = np.argsort(feature_ranges)[-n_features_to_discretize:]
                self.top_features[cluster_id] = top_features
                self.logger.debug(f"top_features cluster{cluster_id}, top features: {top_features}")
                
                # Discretize selected features
                discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='onehot-dense', strategy='quantile')
                X_discrete = discretizer.fit_transform(X_dense[:, top_features])
                new_features.append(scipy.sparse.csr_matrix(X_discrete))
                # feature_descriptions.extend([f"discrete_bin_{i}" for i in range(X_discrete.shape[1])])
                self.discretizer[cluster_id] = discretizer
                self.logger.debug(f"Discretizer cluster{cluster_id}, new feature: {X_discrete.shape}")
                new_feature_num += X_discrete.shape[1]
                # self.logger.debug(f"new_features cluster{cluster_id}, new shape: {len(new_features)}")

                # 3. Feature interactions (for non-zero elements)
                interactions = self.feature_interaction(X_dense, cluster_id)
                new_feature_num += len(interactions)
                self.logger.debug(f"Feature interaction cluster{cluster_id}, new featurenum: {len(interactions)}")
                new_features.extend(interactions)
                self.logger.debug(f"cluster{cluster_id}, new featurenum: {new_feature_num}")

                new_feature_indices.extend([i for i in range(max_feature_index+1, max_feature_index+1+new_feature_num)])
                self.logger.debug(f"new_feature_indices cluster{cluster_id}, new indices: {new_feature_indices}")

                # Combine all new features
                X_new = scipy.sparse.hstack([X] + new_features).tocsr()
                self.logger.debug(f"X_new cluster{cluster_id}, new shape: {X_new.shape}")
                clusters[cluster_id] = (X_new, y, new_feature_indices)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Feature construction failed: {str(e)}")
            return clusters
    
    def transform(self, clusters: Dict) -> Dict:
        """
        Transform existing features
        """
        try:
            for cluster_id, (X, y, feature_indices) in clusters.items():
                self.logger.debug(f"before feature construction for cluster {cluster_id}: {feature_indices}")
                X_dense = X.toarray()
                new_features = []
                max_feature_index = max(feature_indices) if feature_indices else 0
                new_feature_indices = feature_indices.copy()
                new_feature_num = 0

                # 1. Standardization of original features
                X_scaled = self.scaler[cluster_id].transform(X_dense)
                new_features.append(scipy.sparse.csr_matrix(X_scaled))
                self.logger.debug(f"Scaler cluster{cluster_id}, new feature: {X_scaled.shape}")
                new_feature_num += X_scaled.shape[1]

                # 2. Discretization of selected features
                X_discrete = self.discretizer[cluster_id].transform(X_dense[:, self.top_features[cluster_id]])
                new_features.append(scipy.sparse.csr_matrix(X_discrete))
                self.logger.debug(f"Discretizer cluster{cluster_id}, new feature: {X_discrete.shape}")
                new_feature_num += X_discrete.shape[1]
                # 3. Feature interactions
                interactions = self.feature_interaction(X_dense, cluster_id)
                new_features.extend(interactions)  
                self.logger.debug(f"Feature interaction cluster{cluster_id}, new featurenum: {len(interactions)}")
                new_feature_num += len(interactions)
                new_feature_indices.extend([i for i in range(max_feature_index+1, max_feature_index+1+new_feature_num)])
                self.logger.debug(f"new_feature_indices cluster{cluster_id}, new indices: {new_feature_indices}")

                # Combine all new features
                X_new = scipy.sparse.hstack([X] + new_features).tocsr()
                self.logger.debug(f"X_new cluster{cluster_id}, new shape: {X_new.shape}")

                clusters[cluster_id] = (X_new, y, new_feature_indices)
            return clusters 
        except Exception as e:
            self.logger.error(f"Feature transformation failed: {str(e)}")
            return clusters 