import os
import numpy as np
import scipy.sparse
from typing import Tuple, Optional, Dict
from sklearn.datasets import load_svmlight_file

class Dataset:
    """Dataset class for handling LIBSVM format datasets"""
    
    def __init__(self, filepath: str):
        """
        Initialize Dataset object
        
        Args:
            filepath: Full path to the dataset file
        """
        self.filepath = filepath
        
        # Lazy loading for data
        self._X = None
        self._y = None
        
        # Dataset statistics
        self.n_features = None
        self.n_samples = None
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    def load_data(self) -> Tuple[scipy.sparse.csr_matrix, np.ndarray]:
        """
        Load data from file
        
        Returns:
            Tuple[scipy.sparse.csr_matrix, np.ndarray]: (feature matrix, label array)
        """
        if self._X is None or self._y is None:
            self._X, self._y = load_svmlight_file(self.filepath)
            
            # Validate data
            if self._X.shape[0] != len(self._y):
                raise ValueError("Feature matrix and label dimensions do not match")
            
            self.n_features = self._X.shape[1]
            self.n_samples = self._X.shape[0]
        
        return self._X, self._y
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        if self.n_features is None:
            self.load_data()
            
        return {
            "filepath": self.filepath,
            "n_features": self.n_features,
            "n_samples": self.n_samples,
        }

# Example usage
if __name__ == '__main__':
    # Load a specific dataset
    try:
        dataset = Dataset("aloi.scale.libsvm")
        X, y = dataset.load_data()
        stats = dataset.get_stats()
        print("\nDataset statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
    except FileNotFoundError as e:
        print(f"\nError: {e}") 