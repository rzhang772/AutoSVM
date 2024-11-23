import unittest
import os
import sys
import numpy as np
import scipy.sparse
from sklearn.datasets import make_classification

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.feature_processor import FeatureProcessor
from src.logger import Logger

class TestFeatureProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.logger = Logger().get_logger('test')
        
        # Generate synthetic sparse data
        X_dense, y = make_classification(
            n_samples=1000,
            n_features=100,
            n_informative=20,
            n_redundant=10,
            n_classes=2,
            random_state=42
        )
        
        # Convert to sparse matrix with some zero entries
        X_dense[X_dense < 0.1] = 0
        cls.X = scipy.sparse.csr_matrix(X_dense)
        cls.y = y
        
        # Create fake cluster labels
        cls.cluster_labels = np.random.randint(0, 3, size=1000)
        
        cls.logger.info(f"Generated test data: {cls.X.shape}")
    
    def test_chi2_only(self):
        """Test chi-square feature selection only"""
        self.logger.info("\nTesting chi-square feature selection")
        
        processor = FeatureProcessor(
            enable_chi2=True,
            enable_qbsofs=False,
            non_zero_threshold=0.01,
            chi_square_threshold=0.05,
            min_features=10
        )
        
        processed_clusters = processor.process_clusters(self.X, self.y, self.cluster_labels)
        
        # Verify results
        self.assertIsNotNone(processed_clusters)
        for cluster_id, (X_filtered, y_cluster, selected_features) in processed_clusters.items():
            self.assertGreaterEqual(len(selected_features), 10)
            self.assertEqual(X_filtered.shape[1], len(selected_features))
            self.assertEqual(X_filtered.shape[0], len(y_cluster))
    
    def test_qbsofs_only(self):
        """Test QBSOFS feature selection only"""
        self.logger.info("\nTesting QBSOFS feature selection")
        
        processor = FeatureProcessor(
            enable_chi2=False,
            enable_qbsofs=True,
            non_zero_threshold=0.01,
            min_features=10,
            qbsofs_params={
                'n_particles': 10,
                'max_iter': 50,
                'alpha': 0.7,
                'beta': 0.5,
                'n_folds': 3,
                'random_state': 42
            }
        )
        
        processed_clusters = processor.process_clusters(self.X, self.y, self.cluster_labels)
        
        # Verify results
        self.assertIsNotNone(processed_clusters)
        for cluster_id, (X_filtered, y_cluster, selected_features) in processed_clusters.items():
            self.assertGreaterEqual(len(selected_features), 10)
            self.assertEqual(X_filtered.shape[1], len(selected_features))
            self.assertEqual(X_filtered.shape[0], len(y_cluster))
    
    def test_both_methods(self):
        """Test both feature selection methods together"""
        self.logger.info("\nTesting both feature selection methods")
        
        processor = FeatureProcessor(
            enable_chi2=True,
            enable_qbsofs=True,
            non_zero_threshold=0.01,
            chi_square_threshold=0.05,
            min_features=10,
            qbsofs_params={
                'n_particles': 10,
                'max_iter': 50,
                'alpha': 0.7,
                'beta': 0.5,
                'n_folds': 3,
                'random_state': 42
            }
        )
        
        processed_clusters = processor.process_clusters(self.X, self.y, self.cluster_labels)
        
        # Verify results
        self.assertIsNotNone(processed_clusters)
        for cluster_id, (X_filtered, y_cluster, selected_features) in processed_clusters.items():
            self.assertGreaterEqual(len(selected_features), 10)
            self.assertEqual(X_filtered.shape[1], len(selected_features))
            self.assertEqual(X_filtered.shape[0], len(y_cluster))
    
    def test_non_zero_filtering(self):
        """Test non-zero value filtering"""
        self.logger.info("\nTesting non-zero filtering")
        
        # Create data with many zero entries
        X_sparse = self.X.copy()
        X_sparse.data[X_sparse.data < 0.5] = 0
        X_sparse.eliminate_zeros()
        
        processor = FeatureProcessor(
            enable_chi2=False,
            enable_qbsofs=False,
            non_zero_threshold=0.3,  # High threshold to test filtering
            min_features=5
        )
        
        processed_clusters = processor.process_clusters(X_sparse, self.y, self.cluster_labels)
        
        # Verify results
        self.assertIsNotNone(processed_clusters)
        for cluster_id, (X_filtered, y_cluster, selected_features) in processed_clusters.items():
            if X_filtered is not None:  # Some clusters might have no features passing threshold
                non_zero_ratios = np.array((X_filtered != 0).sum(axis=0)).flatten() / X_filtered.shape[0]
                self.assertTrue(all(ratio >= 0.3 for ratio in non_zero_ratios))
    
    def test_error_handling(self):
        """Test error handling with invalid input"""
        self.logger.info("\nTesting error handling")
        
        processor = FeatureProcessor()
        
        # Test with empty matrix
        empty_X = scipy.sparse.csr_matrix((0, 10))
        empty_y = np.array([])
        empty_labels = np.array([])
        
        processed_clusters = processor.process_clusters(empty_X, empty_y, empty_labels)
        self.assertEqual(len(processed_clusters), 0)
        
        # Test with incompatible dimensions
        wrong_y = np.array([0, 1])  # Wrong length
        with self.assertRaises(Exception):
            processor.process_clusters(self.X, wrong_y, self.cluster_labels)

if __name__ == '__main__':
    unittest.main(verbosity=2) 