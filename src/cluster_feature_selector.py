import numpy as np
import scipy.sparse
from typing import List, Tuple
import logging
from sklearn.preprocessing import KBinsDiscretizer

class ClusterFeatureSelector:
    """Class for selecting high entropy features for clustering"""
    
    def __init__(self, select_ratio: float = 0.3, logger: logging.Logger = None):
        """
        Initialize ClusterFeatureSelector
        
        Args:
            select_ratio: Ratio of features to select based on entropy
            logger: Logger instance
        """
        self.select_ratio = select_ratio
        self.logger = logger or logging.getLogger('cluster_feature_selector')
        self.selected_features_ = None  # Store selected feature indices
    
    def fit_transform(self, X: scipy.sparse.csr_matrix, select_strategy = 'bottom') -> Tuple[scipy.sparse.csr_matrix, List[int]]:
        """
        Fit the selector on data and transform it
        
        Args:
            X: Input feature matrix
            
        Returns:
            Tuple of (selected feature matrix, selected feature indices)
        """
        try:
            self.logger.info("Calculating feature entropies...")
            
            # Calculate entropy for each feature
            entropies = []
            for i in range(X.shape[1]):  # 按列迭代
                column = X[:, i]  # 获取稀疏矩阵的第 i 列
                entropy = self._calculate_entropy_sparse(column)
                entropies.append(entropy)

            entropies = np.array(entropies)
            
            # Select top features based on entropy
            n_features_to_select = int(np.ceil(X.shape[1] * self.select_ratio))
            if select_strategy == 'top':
                self.selected_features_ = np.argsort(entropies)[-n_features_to_select:]
            elif select_strategy == 'bottom':
                self.selected_features_ = np.argsort(entropies)[:n_features_to_select]
            else:
                raise ValueError(f"Invalid select_strategy: {select_strategy}")
            
            # Select features
            X_selected = X[:, self.selected_features_]
            
            self.logger.info(f"Selected {len(self.selected_features_)} features based on entropy")
            self.logger.debug(f"Selected feature indices: {self.selected_features_}")
            return X_selected, self.selected_features_.tolist()
            
        except Exception as e:
            self.logger.error(f"Feature selection failed: {str(e)}")
            return X, []
    
    def transform(self, X: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
        """
        Transform data using the selected features from training
        
        Args:
            X: Input feature matrix
            
        Returns:
            Transformed feature matrix
        """
        if self.selected_features_ is None:
            raise ValueError("FeatureSelector not fitted. Call fit_transform first.")
        
        try:
            self.logger.info("Transforming data using selected features...")
            X_selected = X[:, self.selected_features_]
            return X_selected
            
        except Exception as e:
            self.logger.error(f"Data transformation failed: {str(e)}")
            return X
    
    def _calculate_entropy_sparse(self, feature: scipy.sparse.csr_matrix) -> float:
        """Calculate the entropy of a sparse feature"""
        # Convert to dense array for processing
        feature_dense = feature.toarray().flatten()

        # 判断是否为连续特征
        # is_continuous = self._is_continuous_feature(feature_dense)
        
        # 如果是连续特征，则先分箱
        # if is_continuous:
        #     print("该特征为连续值，进行分箱操作...")
        #     feature_dense = self._bin_continuous_feature(feature_dense)
        # else:
        #     print("该特征为离散值，直接计算信息熵...")
        feature_dense = self._bin_continuous_feature(feature_dense)
        feature_dense = feature_dense.astype(int)
        
        # Normalize feature values to probabilities
        value_counts = np.bincount(feature_dense)
        probabilities = value_counts / len(feature_dense)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Add small value to avoid log(0)
        return entropy 
    def _is_continuous_feature(self, feature_dense: np.ndarray) -> bool:
        """
        判断特征是否为连续值。
        :param feature_dense: 密集特征数组
        :return: True 如果为连续值，否则 False
        """
        unique_values = np.unique(feature_dense)
        unique_ratio = len(unique_values) / len(feature_dense)
        return unique_ratio > 0.1

    def _bin_continuous_feature(self, feature_dense: np.ndarray) -> np.ndarray:
        """
        对连续特征进行分箱。
        :param feature_dense: 密集特征数组
        :return: 分箱后的特征数组
        """
        binning = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
        binned_feature = binning.fit_transform(feature_dense.reshape(-1, 1))
        return binned_feature.flatten()