import os
import numpy as np
import scipy.sparse
from datetime import datetime
from sklearn.model_selection import train_test_split
from typing import Dict
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import logging

# Configuration constants
RAW_DATA_DIR = "./data/raw"
OUTPUT_DATA_DIR = "./data/processed"
LOG_DIR = "./log"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Dataset lists
train_test_split_datasets = [
    # "aloi.scale",
    # "cadata",
    # "HIGGS",
    # "news20.binary",
    # "url_combined_normalized",
    "space_ga_scale"
]

train_set_datasets = [
    # "sector.scale",
    # "avazu-app.tr",
    # "epsilon_normalized",
    # "log1p.E2006.train",
    # "YearPredictionMSD",
]

test_set_datasets = [
    # "sector.t.scale",
    # "avazu-app.t",
    # "epsilon_normalized.t",
    # "log1p.E2006.test",
    # "YearPredictionMSD.t",
]

def setup_logging() -> logging.Logger:
    """Configure preprocessing specific logging"""
    # Create log directory if not exists
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    # Create logger
    logger = logging.getLogger('preprocess')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Add file handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, f"preprocess_{timestamp}.log"),
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def analyze_dataset(X: scipy.sparse.csr_matrix, y: np.ndarray) -> Dict:
    """Analyze dataset features"""
    n_samples, n_features = X.shape
    
    # Calculate feature sparsity using sparse matrix operations
    n_nonzero = np.diff(X.indptr)  # Number of non-zeros per row
    sparsity = 1 - (n_nonzero / n_features)
    
    # Calculate label distribution
    unique_labels, label_counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique_labels, label_counts))
    
    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "avg_sparsity": float(sparsity.mean()),
        "label_distribution": class_dist
    }

def generate_report(dataset_stats: Dict[str, Dict]) -> str:
    """Generate dataset analysis report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = f"Dataset Analysis Report - {timestamp}\n"
    report += "=" * 50 + "\n\n"
    
    for dataset_name, stats in dataset_stats.items():
        report += f"Dataset: {dataset_name}\n"
        report += "-" * 30 + "\n"
        report += f"Number of samples: {stats['n_samples']}\n"
        report += f"Number of features: {stats['n_features']}\n"
        report += f"Average sparsity: {stats['avg_sparsity']:.4f}\n"
        report += f"Label distribution: {stats['label_distribution']}\n"
        report += "\n"
    
    return report

def process_dataset(input_path: str, output_dir: str, dataset_name: str, need_split: bool, logger: logging.Logger) -> Dict:
    """Process a single dataset"""
    logger.info(f"Processing dataset: {dataset_name}")
    
    # Load data in sparse format
    X, y = load_svmlight_file(input_path)
    
    # Analyze data
    stats = analyze_dataset(X, y)
    
    if need_split:
        # Split data while maintaining sparsity
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Save split data
        base_name = os.path.splitext(dataset_name)[0]
        train_file = os.path.join(output_dir, f"{base_name}_train.txt")
        test_file = os.path.join(output_dir, f"{base_name}_test.txt")
        
        dump_svmlight_file(X_train, y_train, train_file)
        dump_svmlight_file(X_test, y_test, test_file)
    else:
        # Just copy to output directory with standardized name
        output_file = os.path.join(output_dir, dataset_name)
        dump_svmlight_file(X, y, output_file)
    
    return stats

def process_datasets() -> None:
    """Process all datasets"""
    logger = setup_logging()
    logger.info("Starting data preprocessing...")
    
    dataset_stats = {}
    
    # Create necessary directories
    for data_type in ['clf', 'reg']:
        os.makedirs(os.path.join(OUTPUT_DATA_DIR, data_type), exist_ok=True)
        
    # Process datasets that need splitting
    for dataset_name in train_test_split_datasets:
        for data_type in ['clf', 'reg']:
            input_path = os.path.join(RAW_DATA_DIR, data_type, dataset_name)
            if os.path.exists(input_path):
                try:
                    stats = process_dataset(
                        input_path,
                        os.path.join(OUTPUT_DATA_DIR, data_type),
                        dataset_name,
                        need_split=True,
                        logger=logger
                    )
                    dataset_stats[f"{data_type}/{dataset_name}"] = stats
                except Exception as e:
                    logger.error(f"Error processing {dataset_name}: {str(e)}")
    
    # Process pre-split datasets
    for train_name, test_name in zip(train_set_datasets, test_set_datasets):
        for data_type in ['clf', 'reg']:
            train_path = os.path.join(RAW_DATA_DIR, data_type, train_name)
            test_path = os.path.join(RAW_DATA_DIR, data_type, test_name)
            
            if os.path.exists(train_path) and os.path.exists(test_path):
                try:
                    # Process training set
                    stats = process_dataset(
                        train_path,
                        os.path.join(OUTPUT_DATA_DIR, data_type),
                        train_name,
                        need_split=False,
                        logger=logger
                    )
                    dataset_stats[f"{data_type}/{train_name}"] = stats
                    
                    # Process test set
                    process_dataset(
                        test_path,
                        os.path.join(OUTPUT_DATA_DIR, data_type),
                        test_name,
                        need_split=False,
                        logger=logger
                    )
                except Exception as e:
                    logger.error(f"Error processing {train_name}/{test_name}: {str(e)}")
    
    # Generate and save report
    report = generate_report(dataset_stats)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(LOG_DIR, f"dataset_report_{timestamp}.txt")
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"Dataset analysis report saved to: {report_file}")
    logger.info("All data processing completed!")

if __name__ == '__main__':
    process_datasets()