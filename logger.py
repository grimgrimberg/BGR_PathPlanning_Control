import time
import logging
import logging.config

import csv

def log_timing(file, section, duration):
    print(f"Logging {section}: {duration}")
    with open(file, mode='a') as timing_file:
        writer = csv.writer(timing_file)
        writer.writerow([section, duration])

def visualize_timing_data(csv_path='timing_log.csv'):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # Check if the directory exists
    directory = os.path.dirname(csv_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Check if the file exists
    if not os.path.exists(csv_path):
        # Create the file with a header row
        with open(csv_path, 'w') as file:
            file.write('Section,Duration\n')  # Write header for the CSV
        print(f"File '{csv_path}' created.")

    # Load Timing Data
    data = pd.read_csv(csv_path)

    # Check if the file contains any data beyond the header
    if data.empty:
        print(f"The file '{csv_path}' exists but has no timing data.")
        return

    # Bar Chart
    avg_durations = data.groupby('Section')['Duration'].mean()
    avg_durations.plot(kind='bar')
    plt.title('Average Execution Time by Section')
    plt.xlabel('Section')
    plt.ylabel('Time (s)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



logging_config = {
    'version': 1,
    'handlers': {
        'fileHandler': {
            'class': 'logging.FileHandler',
            'formatter': 'myFormatter',
            'filename': f'logs/{time.ctime()}.log'
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
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger('SimLogger')
    logger.info('Logger initalized')
