import logging
import pickle
import os

def load_pv_model(name="saved_models/Solar_Model_Results.wwm"):
    if not name.endswith(".wwm"):
        logging.info(f"File being loaded is not in the correct format (.wwm): {name}")
        logging.info("*******************")
        return
    try:
        if not os.path.exists(name):
            raise FileNotFoundError(f"The file '{name}' does not exist.")
        with open(name, 'rb') as filehandler_open:
            logging.info(f"Model successfully loaded: {name}")
            logging.info("*******************")
            return pickle.load(filehandler_open)
    except FileNotFoundError as e:
        logging.info(f"Error: {e}")
        logging.info("*******************")
    except Exception as e:
        logging.info(f"Error loading model from '{name}': {e}")
        logging.info("*******************")
        