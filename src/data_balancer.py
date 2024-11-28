import numpy as np
import scipy.sparse
from typing import Tuple, Dict, List
import logging
from collections import Counter

class DataBalancer:
    """Class for balancing class distribution in clustered subsets"""
    
    def __init__(self,
                 min_ratio: float = 0.1,     # Minimum ratio for any class
                 max_samples: int = 100,      # Maximum samples to add per class
                 random_state: int = 42,
                 logger: logging.Logger = None):
        """
        Initialize DataBalancer
        
        Args:
            min_ratio: Minimum ratio required for each class
            max_samples: Maximum number of samples to add per class
            random_state: Random seed
            logger: Logger instance from main
        """
        self.min_ratio = min_ratio
        self.max_samples = max_samples
        self.random_state = random_state
        self.logger = logger or logging.getLogger('data_balancer')
        np.random.seed(random_state)
    
    def balance_clusters(self,
                        X_full: scipy.sparse.csr_matrix,
                        y_full: np.ndarray,
                        processed_clusters: Dict) -> Dict:
        """
        Balance class distribution in each cluster
        
        Args:
            X_full: Full training data matrix
            y_full: Full training labels
            processed_clusters: Dictionary of processed cluster data
            
        Returns:
            Dictionary of balanced cluster data
        """
        balanced_clusters = {}
        
        # Create index map for each class
        class_indices = {label: np.where(y_full == label)[0] 
                        for label in np.unique(y_full)}
        
        for cluster_id, (X_cluster, y_cluster, selected_features) in processed_clusters.items():
            # self.logger.info(f"\nProcessing cluster {cluster_id}")
            
            # Calculate current class distribution
            class_counts = Counter(y_cluster)
            majority_count = max(class_counts.values())
            
            # Log original distribution at debug level
            # self.logger.debug("Original class distribution:")
            # for label, count in class_counts.items():
            #     ratio = count / majority_count
            #     self.logger.debug(f"Class {label}: {count} samples ({ratio:.2%} of majority)")
            
            # Identify underrepresented classes
            underrep_classes = {
                label: count for label, count in class_counts.items()
                if count / majority_count < self.min_ratio
            }
            
            if not underrep_classes:
                self.logger.debug("No class balancing needed")
                balanced_clusters[cluster_id] = (X_cluster, y_cluster, selected_features)
                continue
            
            # self.logger.info(f"Balancing {len(underrep_classes)} classes in cluster {cluster_id}")
            
            # Add samples for underrepresented classes
            additional_X = []
            additional_y = []
            
            for label, count in underrep_classes.items():
                # Calculate how many samples to add based on majority class
                target_count = int(majority_count * self.min_ratio)
                samples_needed = target_count - count
                
                # Get available indices for this class
                available_indices = list(set(class_indices[label]) - 
                                      set(np.where(y_cluster == label)[0]))
                
                if not available_indices:
                    self.logger.debug(f"No additional samples available for class {label}")
                    continue
                
                # Determine actual number of samples to add
                samples_to_add = min(
                    samples_needed,
                    self.max_samples,
                    len(available_indices)
                )
                
                if samples_to_add <= 0:
                    continue
                
                # Randomly select samples with replacement if necessary
                if samples_to_add > len(available_indices):
                    self.logger.debug(f"Using replacement sampling for class {label} "
                                    f"(need {samples_to_add}, have {len(available_indices)})")
                    selected_indices = np.random.choice(
                        available_indices,
                        size=samples_to_add,
                        replace=True
                    )
                else:
                    selected_indices = np.random.choice(
                        available_indices,
                        size=samples_to_add,
                        replace=False
                    )
                
                # Add selected samples
                additional_X.append(X_full[selected_indices])
                additional_y.extend([label] * samples_to_add)
                
                # self.logger.debug(f"Added {samples_to_add} samples for class {label}")
            
            if additional_X:
                # Combine original and additional samples
                X_balanced = scipy.sparse.vstack([X_cluster] + additional_X)
                y_balanced = np.concatenate([y_cluster, additional_y])
                
                # Log final distribution at debug level
                new_counts = Counter(y_balanced)
                new_majority_count = max(new_counts.values())
                # self.logger.debug("\nFinal class distribution:")
                # for label, count in new_counts.items():
                #     ratio = count / new_majority_count
                #     self.logger.debug(f"Class {label}: {count} samples ({ratio:.2%} of majority)")
                
                self.logger.info(f"Cluster {cluster_id} balanced: {len(y_cluster)} -> {len(y_balanced)} samples")
                balanced_clusters[cluster_id] = (X_balanced, y_balanced, selected_features)
            else:
                self.logger.info(f"Cluster {cluster_id} unchanged: {len(y_cluster)} samples")
                balanced_clusters[cluster_id] = (X_cluster, y_cluster, selected_features)
        
        return balanced_clusters 