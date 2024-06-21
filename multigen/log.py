import logging
import threading


def thread_id_filter(record):
    """Inject thread_id to log records"""
    record.thread_id = threading.get_ident()
    return record

def setup_logger(path='log_file.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('log_file.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    ch.addFilter(thread_id_filter)
    fh.addFilter(thread_id_filter)
    formatter = logging.Formatter('%(asctime)s - %(thread)d - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)

