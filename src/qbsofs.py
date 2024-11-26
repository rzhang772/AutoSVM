import numpy as np
import scipy.sparse
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC, SVR
from typing import List, Tuple, Dict
import logging

class QBSOFS:
    """Q-learning Based Bee Swarm Optimization for Feature Selection"""
    
    def __init__(self,
                 task_type: str,
                 n_bees: int = 20,           # Number of bees in colony
                 max_iter: int = 100,         # Maximum iterations
                 n_elite: int = 5,            # Number of elite sites
                 n_best: int = 10,            # Number of best sites
                 n_elite_bees: int = 4,       # Bees recruited for elite sites
                 n_best_bees: int = 2,        # Bees recruited for best sites
                 learning_rate: float = 0.1,  # Q-learning rate
                 discount_factor: float = 0.9, # Q-learning discount factor
                 epsilon: float = 0.1,        # Exploration rate
                 n_folds: int = 5,            # Cross-validation folds
                 random_state: int = 42,
                 logger: logging.Logger = None):
        """
        Initialize QBSOFS
        
        Args:
            task_type: Task type
            n_bees: Number of scout bees
            max_iter: Maximum iterations
            n_elite: Number of elite sites
            n_best: Number of best sites (excluding elite)
            n_elite_bees: Bees recruited for elite sites
            n_best_bees: Bees recruited for best sites
            learning_rate: Q-learning rate
            discount_factor: Q-learning discount factor
            epsilon: Exploration rate
            n_folds: Number of cross-validation folds
            random_state: Random seed
            logger: Logger for logging messages
        """
        self.task_type = task_type
        self.n_bees = n_bees
        self.max_iter = max_iter
        self.n_elite = n_elite
        self.n_best = n_best
        self.n_elite_bees = n_elite_bees
        self.n_best_bees = n_best_bees
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.n_folds = n_folds
        self.random_state = random_state
        self.logger = logger
        
        np.random.seed(random_state)
        self.q_table = {}  # Q-table for storing state-action values
    
    def _get_state_key(self, features: np.ndarray) -> str:
        """Convert feature selection state to string key"""
        return ''.join(map(str, features.astype(int)))
    
    def _get_q_value(self, state: str, action: int) -> float:
        """Get Q-value for state-action pair"""
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        return self.q_table[state][action]
    
    def _update_q_value(self, state: str, action: int, reward: float, next_state: str):
        """Update Q-value using Q-learning update rule"""
        if next_state in self.q_table:
            max_next_q = max(self.q_table[next_state].values())
        else:
            max_next_q = 0.0
            
        current_q = self._get_q_value(state, action)
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state][action] = new_q
    
    def _local_search(self, 
                     X: scipy.sparse.csr_matrix,
                     y: np.ndarray,
                     features: np.ndarray,
                     n_steps: int) -> np.ndarray:
        """Perform Q-learning based local search"""
        current_features = features.copy()
        current_state = self._get_state_key(current_features)
        
        for _ in range(n_steps):
            # ε-greedy action selection
            if np.random.random() < self.epsilon:
                # Exploration: random flip
                action = np.random.randint(len(features))
            else:
                # Exploitation: best Q-value
                state_q_values = self.q_table.get(current_state, {})
                if state_q_values:
                    action = max(state_q_values.items(), key=lambda x: x[1])[0]
                else:
                    action = np.random.randint(len(features))
            
            # Apply action (flip feature)
            next_features = current_features.copy()
            next_features[action] = not next_features[action]
            
            # Calculate reward (improvement in fitness)
            current_fitness = self._fitness(X, y, current_features)
            next_fitness = self._fitness(X, y, next_features)
            reward = next_fitness - current_fitness
            
            # Update Q-value
            next_state = self._get_state_key(next_features)
            self._update_q_value(current_state, action, reward, next_state)
            
            # Move to next state if better
            if next_fitness > current_fitness:
                current_features = next_features
                current_state = next_state
        
        return current_features
    
    def select_features(self, 
                       X: scipy.sparse.csr_matrix,
                       y: np.ndarray,
                       min_features: int) -> Tuple[List[int], float]:
        """
        Select features using BSO with Q-learning
        
        Args:
            X: Feature matrix
            y: Target labels
            min_features: Minimum number of features to select
            
        Returns:
            Tuple of (selected feature indices, best fitness score)
        """
        n_features = X.shape[1]
        best_solution = None
        best_fitness = float('-inf')
        
        # Initialize scout bees randomly
        solutions = [
            np.random.random(n_features) > 0.5
            for _ in range(self.n_bees)
        ]
        
        # Ensure minimum features
        for solution in solutions:
            if np.sum(solution) < min_features:
                zero_indices = np.where(~solution)[0]
                selected = np.random.choice(
                    zero_indices,
                    min_features - np.sum(solution),
                    replace=False
                )
                solution[selected] = True
        
        for iteration in range(self.max_iter):
            self.logger.debug(f"QBSOFS Iteration {iteration + 1}/{self.max_iter}")
            
            # Evaluate all solutions
            fitness_scores = [self._fitness(X, y, s) for s in solutions]
            self.logger.debug(f"Fitness scores: {fitness_scores}")
            
            # Sort solutions by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_indices = sorted_indices[:self.n_elite]
            best_indices = sorted_indices[self.n_elite:self.n_elite + self.n_best]
            
            # Update best solution
            if fitness_scores[sorted_indices[0]] > best_fitness:
                best_solution = solutions[sorted_indices[0]].copy()
                best_fitness = fitness_scores[sorted_indices[0]]
            
            # Create new solutions
            new_solutions = []
            
            # Recruit bees for elite sites
            for idx in elite_indices:
                base_solution = solutions[idx]
                for _ in range(self.n_elite_bees):
                    # Perform Q-learning based local search
                    new_solution = self._local_search(
                        X, y, base_solution.copy(), n_steps=5
                    )
                    new_solutions.append(new_solution)
            
            # Recruit bees for best sites
            for idx in best_indices:
                base_solution = solutions[idx]
                for _ in range(self.n_best_bees):
                    # Perform Q-learning based local search
                    new_solution = self._local_search(
                        X, y, base_solution.copy(), n_steps=3
                    )
                    new_solutions.append(new_solution)
            self.logger.debug(f"New solutions bees: {new_solutions}")
            
            # Scout bees explore randomly
            n_scouts = self.n_bees - len(new_solutions)
            for _ in range(n_scouts):
                scout_solution = np.random.random(n_features) > 0.5
                if np.sum(scout_solution) < min_features:
                    zero_indices = np.where(~scout_solution)[0]
                    selected = np.random.choice(
                        zero_indices,
                        min_features - np.sum(scout_solution),
                        replace=False
                    )
                    scout_solution[selected] = True
                new_solutions.append(scout_solution)
            self.logger.debug(f"New solutions scouts: {new_solutions}")
            # Update population
            solutions = new_solutions
            
            self.logger.debug(f"Best fitness: {best_fitness:.4f}, "
                           f"Selected features: {np.sum(best_solution)}")
        
        selected_features = np.where(best_solution)[0]
        return best_solution, best_fitness
    
    def _fitness(self, X: scipy.sparse.csr_matrix, y: np.ndarray, solution: np.ndarray) -> float:
        """Calculate fitness using SVM with cross-validation"""
        selected = np.where(solution)[0]
        if len(selected) == 0:
            return float('-inf')
            
        try:
            X_selected = X[:, selected]
            if self.task_type == 'clf'  :
                clf = SVC(kernel='linear', random_state=self.random_state)
            elif self.task_type == 'reg':
                clf = SVR(random_state=self.random_state)
            scores = cross_val_score(clf, X_selected, y, cv=self.n_folds)
            return np.mean(scores)
        except Exception as e:
            self.logger.error(f"Fitness calculation failed: {str(e)}")
            return float('-inf') 