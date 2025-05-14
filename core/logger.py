import time
import logging
import logging.config
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
    return f'logs/{timestamp}.log'

# Updated logging configuration
logging_config = {
    'version': 1,
    'handlers': {
        'fileHandler': {
            'class': 'logging.FileHandler',
            'formatter': 'myFormatter',
            'filename': get_safe_log_filename(),  # Use the safe filename function
        }
    },
    'loggers': {
        'SimLogger': {
            'handlers': ['fileHandler'],
            'level': 'INFO',
        }
    },
    'formatters': {
        'myFormatter': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }
}

def init_logger():
    import os

    # Ensure the logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Initialize the logger
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger('SimLogger')
    logger.info('Logger initialized')

def log_timing(section, duration):
    global timer
    # Append the new data as a row
    new_row = {'Section': section, 'Duration': duration}
    timer = pd.concat([timer, pd.DataFrame([new_row])], ignore_index=True)

