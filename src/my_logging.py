import logging
import os 
from datetime import datetime

log_file=f"{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}.log "
log_path=os.path.join(os.getcwd(),'loggers')
os.makedirs(log_path, exist_ok=True)

file=os.path.join(log_path, log_file)

logging.basicConfig(filename=file,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
