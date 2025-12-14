import os
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = os.getenv('DATA_DIR')
TRAINING_LOG_FILE = os.getenv('TRAINING_LOG_FILE')