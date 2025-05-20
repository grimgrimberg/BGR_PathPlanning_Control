import time
import logging
import logging.config
import logging.handlers
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

timer = pd.DataFrame(columns=['Section', 'Duration'])


def visualize_timing_data():
    global timer
    # Bar Chart
    avg_durations = timer.groupby('Section')['Duration'].mean()
    avg_durations.plot(kind='bar')
    plt.title('Average Execution Time by Section')
    plt.xlabel('Section')
    plt.ylabel('Time (s)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Create a safe filename for the log file
def get_safe_log_filename():
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS
    return f'output/logs/formula_{timestamp}.log'

# Enhanced logging configuration
logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s | %(name)-12s | %(levelname)-8s | %(processName)s-%(thread)d | %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'simple': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        },
        'file_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': get_safe_log_filename(),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        },
        'error_file_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': f'output/logs/formula_errors_{time.strftime("%Y-%m-%d_%H-%M-%S")}.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        },
        'performance_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'detailed',
            'filename': f'output/logs/formula_performance_{time.strftime("%Y-%m-%d_%H-%M-%S") }.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 3
        }
    },
    'loggers': {
        'SimLogger': {
            'level': 'DEBUG',
            'handlers': ['console', 'file_handler', 'error_file_handler'],
            'propagate': False
        },
        'PerformanceLogger': {
            'level': 'INFO',
            'handlers': ['performance_handler', 'console'],
            'propagate': False
        },
        'Controller': {
            'level': 'DEBUG',
            'handlers': ['file_handler', 'console'],
            'propagate': False
        },
        'Planner': {
            'level': 'DEBUG',
            'handlers': ['file_handler', 'console'],
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file_handler']
    }
}

def init_logger():
    """Initialize the enhanced logging system."""
    # Ensure the logs directory exists
    os.makedirs("output/logs", exist_ok=True)
    
    # Initialize the logger with the enhanced configuration
    logging.config.dictConfig(logging_config)
    
    # Get the main logger
    logger = logging.getLogger('SimLogger')
    logger.info('Enhanced logging system initialized')
    
    # Initialize the performance logger
    perf_logger = logging.getLogger('PerformanceLogger')
    perf_logger.info('Performance logging initialized')

def log_timing(section, duration):
    """Log timing information with enhanced detail."""
    global timer
    # Log to the performance logger
    perf_logger = logging.getLogger('PerformanceLogger')
    perf_logger.info(f"Section timing - {section}: {duration:.4f} seconds")
    
    # Store in the DataFrame for visualization
    new_row = {'Section': section, 'Duration': duration}
    timer = pd.concat([timer, pd.DataFrame([new_row])], ignore_index=True)

