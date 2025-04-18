"""
Logging utilities for the Continuous Learning Lab.
"""
import json
import logging
import logging.handlers
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Configuration
LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
METRICS_FILE = os.getenv("METRICS_FILE", "logs/metrics.tsv")
RESPONSES_DIR = os.getenv("RESPONSES_DIR", "responses")

# Ensure directories exist
Path(LOG_FILE).parent.mkdir(exist_ok=True, parents=True)
Path(METRICS_FILE).parent.mkdir(exist_ok=True, parents=True)
Path(RESPONSES_DIR).mkdir(exist_ok=True, parents=True)

# Thread safety for file operations
_file_lock = threading.Lock()

# Configure main logger
logger = logging.getLogger("continuous_learning")
if not logger.handlers:
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
    )
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO")))

def log_event(event_type: str, **kwargs: Any) -> None:
    """
    Log a structured event with timestamp and additional data.
    
    Args:
        event_type: Type of event being logged
        **kwargs: Additional data to include in the log
    """
    logger.info(json.dumps({
        "event": event_type,
        "timestamp": time.time(),
        **kwargs
    }))

def save_metric(update_index: int, acceptance_rate: float, 
                mean_score: float, threshold: int) -> None:
    """
    Save performance metrics to the metrics file.
    
    Args:
        update_index: Incremental index of model updates
        acceptance_rate: Proportion of samples above threshold
        mean_score: Average score of recent samples
        threshold: Current acceptance threshold
    """
    with _file_lock:
        df = pd.DataFrame([[update_index, acceptance_rate, mean_score, threshold]], 
                         columns=["update_index", "acceptance_rate", "mean_score", "threshold"])
        
        file_exists = os.path.exists(METRICS_FILE)
        df.to_csv(
            METRICS_FILE, 
            sep="\t", 
            index=False, 
            mode="a" if file_exists else "w",
            header=not file_exists
        )

def save_teacher_response(
    qid: str, 
    model: str, 
    prompt: str, 
    response: str, 
    score: Optional[int] = None,
    is_correction: bool = False
) -> str:
    """
    Save teacher's raw response (both grading and corrections).
    
    Args:
        qid: Question unique identifier
        model: Teacher model name
        prompt: Original prompt
        response: Raw response from the teacher
        score: Numeric score (if this was a grading response)
        is_correction: Whether this is a correction response
    
    Returns:
        Path to the saved response file
    """
    # Sanitize model name for filename
    safe_model = model.replace("/", "_").replace(":", "_")
    response_type = "correction" if is_correction else "grade"
    
    filename = f"{RESPONSES_DIR}/{qid}_{safe_model}_{response_type}.json"
    
    with _file_lock:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "qid": qid,
                "model": model,
                "type": response_type,
                "score": score,
                "prompt": prompt,
                "response": response,
                "timestamp": time.time()
            }, f, indent=2, ensure_ascii=False)
    
    log_event(
        "teacher_response_saved", 
        qid=qid, 
        model=model, 
        type=response_type,
        file=filename
    )
    return filename
