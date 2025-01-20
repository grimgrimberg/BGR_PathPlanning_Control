import time
import logging
import logging.config
import csv
import os




def visualize_timing_data(csv_path='timing_log.csv'):
    import pandas as pd
    import matplotlib.pyplot as plt

    # Check if the file exists
    if not os.path.exists(csv_path):
        print(f"The file '{csv_path}' does not exist. Creating a new file with headers.")
        # Create the file with a header row
        with open(csv_path, 'w') as file:
            file.write('Section,Duration\n')  # Write header
        return

    # Load Timing Data
    data = pd.read_csv(csv_path)

    # Ensure there is data beyond the header
    if data.shape[0] <= 1:  # If only the header exists
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

def log_timing(file, section, duration):

    # Check if the file exists
    if not os.path.exists(file):
        # Create the file with a header row
        with open(file, 'w', newline='') as timing_file:
            writer = csv.writer(timing_file)
            writer.writerow(['Section', 'Duration'])  # Write the header

    # Append the timing data
    with open(file, 'a', newline='') as timing_file:
        writer = csv.writer(timing_file)
        writer.writerow([section, duration])


