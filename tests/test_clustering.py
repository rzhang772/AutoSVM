import unittest
import os
import sys
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
from src.dataset import Dataset
from src.svmcluster import SVMCluster, ClusterMethod
from src.logger import Logger

# Test datasets arrays (now using full paths)
CLF_DATASETS = [
    # "data/processed/clf/aloi_train.txt",
    # "data/processed/clf/sector.scale",
    "data/processed/clf/avazu-app.tr",
    # "data/processed/clf/epsilon_normalized",
    # "data/processed/clf/HIGGS_train.txt",
]

REG_DATASETS = [
    # "data/processed/reg/cadata_train.txt",
    # "data/processed/reg/log1p.E2006.train",
    # "data/processed/reg/YearPredictionMSD",
]

class TestClustering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.logger = Logger().get_logger('test')
        
        # Set up paths
        cls.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.output_dir = os.path.join(cls.project_root, "output", "test_cluster")
        
        # Create output directory if it doesn't exist
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Verify datasets exist
        cls.verify_datasets()
        
        # Initialize clusterer
        cls.clusterer = SVMCluster(
            output_dir=cls.output_dir,
            enable_sampling=True,
            sample_ratio=0.1
        )
    
    @classmethod
    def verify_datasets(cls):
        """Verify all test datasets exist"""
        for filepath in CLF_DATASETS + REG_DATASETS:
            try:
                full_path = os.path.join(cls.project_root, filepath)
                dataset = Dataset(full_path)
                X, y = dataset.load_data()
                cls.logger.info(f"Verified dataset: {filepath}, shape: {X.shape}")
            except Exception as e:
                cls.logger.error(f"Failed to load dataset {filepath}: {str(e)}")
                raise

    def test_all_datasets(self):
        """Test all clustering methods on all datasets"""
        self.logger.info("\n=== Starting tests for all datasets ===")
        
        # Test all datasets
        for filepath in CLF_DATASETS + REG_DATASETS:
            try:
                full_path = os.path.join(self.project_root, filepath)
                dataset = Dataset(full_path)
                X, y = dataset.load_data()
                self.logger.info(f"\nTesting dataset: {filepath}")
                self.logger.info(f"Data shape: {X.shape}")
                
                # Test all clustering methods
                self._test_kmeans_optimal_k_silhouette(X, filepath)
                self._test_kmeans_optimal_k_gap(X, filepath)
                self._test_kmeans_fixed_k(X, filepath)
                self._test_random_clustering(X, filepath)
                self._test_fifo_clustering(X, filepath)
                # self._test_invalid_combinations(X, filepath)
                
            except Exception as e:
                self.logger.error(f"Failed testing dataset {filepath}: {str(e)}")
                raise

    def _test_kmeans_optimal_k_silhouette(self, X, dataset_name):
        """Test k-means with silhouette method"""
        self.logger.info(f"Testing silhouette method on {dataset_name}")
        try:
            best_k, results, labels, model = self.clusterer.cluster(
                X,
                k_range=range(5, 8),
                method="silhouette",
                algorithm="kmeans"
            )
            
            # Verify results
            self.assertIsNotNone(best_k)
            self.assertTrue(5 <= best_k <= 7)
            self.assertIn("scores", results)
            self.assertEqual(len(np.unique(labels)), best_k)
            
            # Verify scores are in valid range [-1, 1]
            self.assertTrue(all(-1 <= score <= 1 for score in results["scores"]))
            
            self.logger.info(f"Silhouette method found best k = {best_k}")
            self.logger.info(f"Silhouette scores: {results['scores']}")
            
        except Exception as e:
            self.logger.error(f"Silhouette method failed on {dataset_name}: {str(e)}")
            raise

    def _test_kmeans_optimal_k_gap(self, X, dataset_name):
        """Test k-means with GAP method"""
        self.logger.info(f"Testing GAP method on {dataset_name}")
        try:
            best_k, results, labels, model = self.clusterer.cluster(
                X,
                k_range=range(5, 8),
                method="gap",
                algorithm="kmeans"
            )
            
            # Verify results
            self.assertIsNotNone(best_k)
            self.assertTrue(5 <= best_k <= 7)
            self.assertIn("gap_values", results)
            self.assertIn("sk_values", results)
            self.assertEqual(len(np.unique(labels)), best_k)
            
            # Verify gap values and sk values exist
            self.assertEqual(len(results["gap_values"]), len(range(5, 8)))
            self.assertEqual(len(results["sk_values"]), len(range(5, 8)))
            
            # Verify sk values are positive (standard errors should always be positive)
            self.assertTrue(all(sk > 0 for sk in results["sk_values"]))
            
            self.logger.info(f"GAP method found best k = {best_k}")
            self.logger.info(f"GAP values: {results['gap_values']}")
            self.logger.info(f"SK values: {results['sk_values']}")
            
        except Exception as e:
            self.logger.error(f"GAP method failed on {dataset_name}: {str(e)}")
            raise

    def _test_kmeans_fixed_k(self, X, dataset_name):
        """Test k-means with fixed k"""
        self.logger.info(f"Testing k-means with fixed k on {dataset_name}")
        fixed_k = 5
        try:
            k, results, labels, model = self.clusterer.cluster(
                X,
                k=fixed_k,
                algorithm="kmeans"
            )
            
            # Verify results
            self.assertEqual(k, fixed_k)
            self.assertEqual(len(np.unique(labels)), fixed_k)
            
            # Verify cluster assignments
            unique, counts = np.unique(labels, return_counts=True)
            self.logger.info(f"Cluster sizes: {counts}")
            
            # Verify model attributes
            self.assertTrue(hasattr(model, 'cluster_centers_'))
            self.assertEqual(model.cluster_centers_.shape[0], fixed_k)
            
            self.logger.info(f"K-means with fixed k={fixed_k} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Fixed k k-means test failed on {dataset_name}: {str(e)}")
            raise

    def _test_random_clustering(self, X, dataset_name):
        """Test random clustering"""
        self.logger.info(f"Testing random clustering on {dataset_name}")
        fixed_k = 5
        try:
            k, results, labels, model = self.clusterer.cluster(
                X,
                k=fixed_k,
                algorithm="random"
            )
            
            # Verify results
            self.assertEqual(k, fixed_k)
            self.assertEqual(len(np.unique(labels)), fixed_k)
            
            # Verify cluster sizes are roughly equal
            unique, counts = np.unique(labels, return_counts=True)
            expected_size = len(labels) // fixed_k
            for count in counts:
                self.assertTrue(abs(count - expected_size) <= 1)
                
            self.logger.info(f"Random clustering completed successfully")
            self.logger.info(f"Cluster sizes: {counts}")
            
        except Exception as e:
            self.logger.error(f"Random clustering test failed on {dataset_name}: {str(e)}")
            raise

    def _test_fifo_clustering(self, X, dataset_name):
        """Test FIFO clustering"""
        self.logger.info(f"Testing FIFO clustering on {dataset_name}")
        fixed_k = 5
        try:
            k, results, labels, model = self.clusterer.cluster(
                X,
                k=fixed_k,
                algorithm="fifo"
            )
            
            # Verify results
            self.assertEqual(k, fixed_k)
            self.assertEqual(len(np.unique(labels)), fixed_k)
            
            # Verify sequential assignment
            first_k_labels = labels[:fixed_k]
            self.assertTrue(np.array_equal(first_k_labels, np.arange(fixed_k)))
            
            # Verify cluster sizes
            unique, counts = np.unique(labels, return_counts=True)
            self.logger.info(f"Cluster sizes: {counts}")
            
            self.logger.info(f"FIFO clustering completed successfully")
            
        except Exception as e:
            self.logger.error(f"FIFO clustering test failed on {dataset_name}: {str(e)}")
            raise

if __name__ == '__main__':
    unittest.main(verbosity=2) 