import os
import logging

logger = logging.getLogger('DOCRE')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

def set_log_dir(log_dir, log_file):
    try:
        if not(os.path.isdir(log_dir)):
            os.makedirs(log_dir)
        
        log_path = os.path.join(log_dir, log_file)

        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as error:
        logger.error(error)
        
        
    


