import logging
from logging.handlers import RotatingFileHandler
import os
from .. import config

LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
APP_NAME = config.PROJECT_NAME
LOG_DIR = "/data/log";
LOG_FILE_INFO  = f'{LOG_DIR}/{APP_NAME}/{APP_NAME}.log'
LOG_FILE_ERROR = f'{LOG_DIR}/{APP_NAME}/{APP_NAME}_error.log'
# LOG_FILE_INFO  = 'log.log'
# LOG_FILE_ERROR = 'log.err'

def setup_log_unicorn():
    # file_handler_info = logging.FileHandler(LOG_FILE_INFO, mode='a')
    # uvicorn_info = logging.getLogger("uvicorn.log")
    # uvicorn_info.addHandler(file_handler_info)

    file_handler_error = RotatingFileHandler(LOG_FILE_INFO, mode='a', maxBytes=20*1024*1024, backupCount=20, encoding='utf-8', delay=0)
    uvicorn_error = logging.getLogger("uvicorn.error")
    uvicorn_error.addHandler(file_handler_error)

    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.disabled = True
    pass

def get_logger(log_name = ''):
    os.makedirs(f'{LOG_DIR}/{APP_NAME}', exist_ok=True)

    log           = logging.getLogger(log_name)
    log_formatter = logging.Formatter(LOG_FORMAT)

    file_handler_info = RotatingFileHandler(LOG_FILE_INFO, mode='a', maxBytes=20*1024*1024, backupCount=20, encoding='utf-8', delay=0)
    file_handler_info.setFormatter(log_formatter)
    file_handler_info.setLevel(logging.INFO)
    log.addHandler(file_handler_info)

    file_handler_error = RotatingFileHandler(LOG_FILE_ERROR, mode='a', maxBytes=20*1024*1024, backupCount=20, encoding='utf-8', delay=0)
    file_handler_error.setFormatter(log_formatter)
    file_handler_error.setLevel(logging.ERROR)
    log.addHandler(file_handler_error)

    log.setLevel(logging.INFO)

    return log