"""Console logging; optional files are created only on explicit request."""
import logging
import os
import sys
import time

def setup_logger(log_dir="logs", log_prefix="graph_build"):
    logger = logging.getLogger(log_prefix)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if os.getenv("CKESWIN_FILE_LOGS") == "1":
        os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(log_dir, log_prefix + time.strftime("_%Y%m%d_%H%M%S.log")), encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
