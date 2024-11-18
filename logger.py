import time
import logging
import logging.config

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
