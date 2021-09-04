"""
Definitions we want to share with other files
"""
from pathlib import Path
import dotenv
import os

ROOT_DIR = str(Path(__file__).resolve().parents[0])
dotenv_path = os.path.join(ROOT_DIR, '.env')
dotenv.load_dotenv(dotenv_path)

DATA_DIR = os.getenv("DATA_DIR")
EXTERNAL_DATA_DIR = os.getenv("EXTERNAL_DATA_DIR")
MODEL_DIR = os.getenv("MODEL_DIR")
TENSORBOARD_DIR = os.getenv("TENSORBOARD_DIR")
RESULTS_DIR = os.getenv("RESULTS_DIR")
MONGO_STR = "bigtop:27017"
MONGO_DB = "experiments"

