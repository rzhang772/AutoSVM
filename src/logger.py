import logging
import os
from datetime import datetime

class Logger:
    """Logger class for handling application logging"""
    
    def __init__(self):
        """Initialize logger"""
        self.logger = None
        
    def get_logger(self, name: str, filename: str = None, level: str = 'INFO') -> logging.Logger:
        """Get logger instance with specified configuration
        
        Args:
            name: Logger name
            filename: Log file path (optional)
            level: Logging level (default: INFO)
            
        Returns:
            Configured logger instance
        """
        if self.logger is not None:
            return self.logger
        
        # Create logger
        self.logger = logging.getLogger(name)
        
        # Convert level string to logging constant
        log_level = getattr(logging, level.upper())
        self.logger.setLevel(log_level)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        error_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # Create file handler for all logs
        if filename:
            file_handler = logging.FileHandler(filename)
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(log_level)
            self.logger.addHandler(file_handler)
        
        # Create console handler only for errors
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(error_formatter)
        console_handler.setLevel(logging.ERROR)  # Only show ERROR and above
        self.logger.addHandler(console_handler)
        
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