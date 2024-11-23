import numpy as np
import scipy.sparse
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from typing import List, Tuple
import logging

class QBSOFS:
    """Quantum-Behaved Particle Swarm Optimization for Feature Selection"""
    
    def __init__(self,
                 n_particles: int = 20,
                 max_iter: int = 100,
                 alpha: float = 0.7,
                 beta: float = 0.5,
                 n_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize QBSOFS
        
        Args:
            n_particles: Number of particles in swarm
            max_iter: Maximum number of iterations
            alpha: Cognitive coefficient
            beta: Social coefficient
            n_folds: Number of cross-validation folds
            random_state: Random seed
        """
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.n_folds = n_folds
        self.random_state = random_state
        self.logger = logging.getLogger('qbsofs')
        
        np.random.seed(random_state)
    
    def select_features(self, 
                       X: scipy.sparse.csr_matrix,
                       y: np.ndarray,
                       min_features: int) -> Tuple[List[int], float]:
        """
        Select features using QBSOFS
        
        Args:
            X: Feature matrix
            y: Target labels
            min_features: Minimum number of features to select
            
        Returns:
            Tuple of (selected feature indices, best fitness score)
        """
        n_features = X.shape[1]
        
        # Initialize particles
        particles = np.random.random((self.n_particles, n_features)) > 0.5
        velocities = np.random.random((self.n_particles, n_features))
        
        # Initialize best positions
        pbest = particles.copy()
        pbest_fitness = np.array([self._fitness(X, y, p) for p in pbest])
        
        gbest = pbest[np.argmax(pbest_fitness)]
        gbest_fitness = np.max(pbest_fitness)
        
        # Main loop
        for iteration in range(self.max_iter):
            self.logger.info(f"QBSOFS Iteration {iteration + 1}/{self.max_iter}")
            
            for i in range(self.n_particles):
                # Update quantum position
                mean_best = (pbest[i] | gbest)
                phi = np.random.random(n_features)
                u = np.random.random(n_features)
                
                # Quantum position update using XOR for difference
                diff = np.logical_xor(gbest, particles[i])
                particles[i] = np.logical_xor(
                    mean_best,
                    (diff & (np.log(1/u) * (2 * phi - 1) > 0))
                )
                
                # Ensure minimum features
                if np.sum(particles[i]) < min_features:
                    zero_indices = np.where(~particles[i])[0]
                    selected = np.random.choice(
                        zero_indices,
                        min_features - np.sum(particles[i]),
                        replace=False
                    )
                    particles[i][selected] = True
                
                # Update personal best
                fitness = self._fitness(X, y, particles[i])
                if fitness > pbest_fitness[i]:
                    pbest[i] = particles[i].copy()
                    pbest_fitness[i] = fitness
                    
                    # Update global best
                    if fitness > gbest_fitness:
                        gbest = particles[i].copy()
                        gbest_fitness = fitness
            
            self.logger.info(f"Best fitness: {gbest_fitness:.4f}, "
                           f"Selected features: {np.sum(gbest)}")
        
        selected_features = np.where(gbest)[0]
        return selected_features.tolist(), gbest_fitness
    
    def _fitness(self, X: scipy.sparse.csr_matrix, y: np.ndarray, particle: np.ndarray) -> float:
        """Calculate fitness using SVM with cross-validation"""
        selected = np.where(particle)[0]
        if len(selected) == 0:
            return 0.0
            
        try:
            X_selected = X[:, selected]
            clf = SVC(kernel='linear', random_state=self.random_state)
            scores = cross_val_score(clf, X_selected, y, cv=self.n_folds)
            return np.mean(scores)
        except Exception as e:
            self.logger.error(f"Fitness calculation failed: {str(e)}")
            return 0.0 