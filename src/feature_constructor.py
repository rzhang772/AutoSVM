import numpy as np
import scipy.sparse
from typing import Tuple, List
import logging
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

class FeatureConstructor:
    """Class for constructing new features from existing ones"""
    
    def __init__(self, n_bins: int = 5):
        """
        Initialize FeatureConstructor
        
        Args:
            n_bins: Number of bins for discretization
        """
        self.n_bins = n_bins
        self.logger = logging.getLogger('feature_constructor')
        
    def construct_features(self, X: scipy.sparse.csr_matrix) -> Tuple[scipy.sparse.csr_matrix, List[str]]:
        """
        Construct new features from input matrix
        
        Args:
            X: Input sparse feature matrix
            
        Returns:
            Tuple of (augmented feature matrix, list of feature descriptions)
        """
        self.logger.info("Starting feature construction")
        feature_descriptions = []
        new_features = []
        
        # Convert to dense for numerical operations
        X_dense = X.toarray()
        
        try:
            # 1. Standardization of original features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_dense)
            new_features.append(scipy.sparse.csr_matrix(X_scaled))
            feature_descriptions.extend([f"scaled_{i}" for i in range(X.shape[1])])
            
            # 2. Discretization of original features
            discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='onehot-dense', strategy='quantile')
            X_discrete = discretizer.fit_transform(X_dense)
            new_features.append(scipy.sparse.csr_matrix(X_discrete))
            feature_descriptions.extend([f"discrete_bin_{i}" for i in range(X_discrete.shape[1])])
            
            # 3. Feature interactions (for non-zero elements)
            for i in range(min(5, X.shape[1])):  # Limit to first 5 features to avoid explosion
                for j in range(i+1, min(6, X.shape[1])):
                    # Multiplication
                    mult = X_dense[:, i] * X_dense[:, j]
                    new_features.append(scipy.sparse.csr_matrix(mult.reshape(-1, 1)))
                    feature_descriptions.append(f"mult_{i}_{j}")
                    
                    # Division (safe)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        div = np.where(X_dense[:, j] != 0, 
                                     X_dense[:, i] / X_dense[:, j],
                                     0)
                    div = np.nan_to_num(div)
                    new_features.append(scipy.sparse.csr_matrix(div.reshape(-1, 1)))
                    feature_descriptions.append(f"div_{i}_{j}")
            
            # 4. Basic statistical features
            # Mean of non-zero elements per row
            row_means = np.array([
                np.mean(row[row != 0]) if np.any(row != 0) else 0 
                for row in X_dense
            ]).reshape(-1, 1)
            new_features.append(scipy.sparse.csr_matrix(row_means))
            feature_descriptions.append("row_mean_nonzero")
            
            # Standard deviation of non-zero elements per row
            row_stds = np.array([
                np.std(row[row != 0]) if np.any(row != 0) else 0 
                for row in X_dense
            ]).reshape(-1, 1)
            new_features.append(scipy.sparse.csr_matrix(row_stds))
            feature_descriptions.append("row_std_nonzero")
            
            # Combine all new features
            X_new = scipy.sparse.hstack([X] + new_features)
            
            self.logger.info(f"Constructed {X_new.shape[1] - X.shape[1]} new features")
            return X_new, feature_descriptions
            
        except Exception as e:
            self.logger.error(f"Feature construction failed: {str(e)}")
            return X, [] 