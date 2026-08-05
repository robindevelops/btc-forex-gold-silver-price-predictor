import logging
import sys
import os

def setup_logging(level='INFO', log_file=None):
    """
    Setup centralized logging configuration.
    
    Args:
        level (str): Logging level.
        log_file (str, optional): Path to log file.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=handlers,
        force=True
    )

def get_logger(name):
    """
    Get a configured logger instance.
    
    Args:
        name (str): Logger name.
        
    Returns:
        logging.Logger: Configured logger.
    """
    return logging.getLogger(name)
