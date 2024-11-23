import os
import logging
from datetime import datetime
from typing import Optional

class Logger:
    """Centralized logging management class"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.logger = None
    
    def get_logger(self, name: str, filename: Optional[str] = None) -> logging.Logger:
        """
        Get logger instance
        
        Args:
            name: Logger name
            filename: Optional custom log filename
        """
        if self.logger is None:
            self.logger = logging.getLogger(name)
            self.logger.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            
            # Create file handler with custom filename if provided
            if filename:
                file_handler = logging.FileHandler(filename)
            else:
                file_handler = logging.FileHandler(f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # Prevent logging to console
            self.logger.propagate = False
        
        return self.logger
    
    def get_all_loggers(self):
        """Get dictionary of all created loggers"""
        return self._loggers

# Example usage
if __name__ == '__main__':
    # Create logger instance
    logger_manager = Logger()
    
    # Get different loggers
    preprocess_logger = logger_manager.get_logger('preprocess')
    cluster_logger = logger_manager.get_logger('cluster')
    
    # Log some messages
    preprocess_logger.info('Starting data preprocessing...')
    cluster_logger.info('Starting clustering analysis...')
    
    # Get same logger again (will return existing one)
    preprocess_logger2 = logger_manager.get_logger('preprocess')
    preprocess_logger2.info('This uses the same logger instance') 