import numpy as np
import scipy.sparse
from sklearn.preprocessing import normalize
from typing import Tuple
import logging

class DataNormalizer:
    """Class for normalizing data using L2 normalization"""
    
    def __init__(self, norm: str = 'l2', logger: logging.Logger = None):
        """
        Initialize DataNormalizer
        
        Args:
            norm: Type of normalization ('l1', 'l2', or 'max')
            logger: Logger instance from main
        """
        self.norm = norm
        self.logger = logger or logging.getLogger('normalizer')
        self.feature_norms_ = None  # Store feature norms from training data
    
    def fit_transform(self, X: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
        """
        Fit normalizer on training data and transform it
        
        Args:
            X: Training data feature matrix
            
        Returns:
            Normalized training feature matrix
        """
        try:
            self.logger.debug(f"Fitting and transforming training data using {self.norm.upper()} normalization...")
            
            # Calculate norms for each feature
            if self.norm == 'l2':
                self.feature_norms_ = np.sqrt(X.multiply(X).sum(axis=1)).A1
            elif self.norm == 'l1':
                self.feature_norms_ = np.abs(X).sum(axis=1).A1
            else:  # max
                self.feature_norms_ = np.abs(X).max(axis=1).A1
            
            # Avoid division by zero
            self.feature_norms_[self.feature_norms_ == 0] = 1.0
            
            # Normalize training data
            X_normalized = X.multiply(1.0 / self.feature_norms_).tocsr()
            
            return X_normalized
            
        except Exception as e:
            self.logger.error(f"Training data normalization failed: {str(e)}")
            return X
    
    def transform(self, X: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
        """
        Transform test data using normalization from training
        
        Args:
            X: Test data feature matrix
            
        Returns:
            Normalized test feature matrix
        """
        if self.feature_norms_ is None:
            raise ValueError("Normalizer not fitted. Call fit_transform first.")
            
        try:
            self.logger.debug(f"Transforming test data using {self.norm.upper()} normalization...")
            
            # Normalize test data using training feature norms
            X_normalized = X.multiply(1.0 / self.feature_norms_).tocsr()
            
            return X_normalized
            
        except Exception as e:
            self.logger.error(f"Test data normalization failed: {str(e)}")
            return X

