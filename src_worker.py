"""
Celery worker for evaluating responses and applying continuous learning.
"""
import os
import time
import random
import statistics
import json
from collections import deque
from typing import Tuple, List, Dict, Any, Optional

import backoff
import openai
import requests
from celery import Celery
from dotenv import load_dotenv

from src_model import train_lora
from src_logger import log_event, save_metric, save_teacher_response

# Load environment variables
load_dotenv()

# Configuration
TEACHER_MODELS = list(filter(None, os.getenv("TEACHER_MODELS", "").split(",")))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Learning parameters
THRESHOLD = int(os.getenv("THRESHOLD_START", "7"))
MIN_THRESHOLD = int(os.getenv("MIN_THRESHOLD", "5"))
BATCH_SIZE = int(os.getenv("GOOD_BATCH_SIZE", "64"))
FORCE_UPDATE_SECS = int(os.getenv("FORCE_UPDATE_SECS", "1800"))
WINDOW_SIZE = int(os.getenv("LOW_PASS_WINDOW", "200"))
RATE_THRESHOLD = float(os.getenv("LOW_PASS_RATE", "0.20"))

# Training parameters
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "4"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-5"))
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "1"))

# State tracking
replay_buffer = deque(maxlen=8000)
score_window = deque(maxlen=WINDOW_SIZE)
last_update_time = time.time()
update_counter = 0

# Create Celery app
celery_app = Celery("tasks", broker=REDIS_URL)

@backoff.on_exception(
    backoff.expo,
    (requests.RequestException, TimeoutError),
    max_tries=3,
    jitter=backoff.full_jitter
)
def call_openrouter_api(
    model: str, 
    messages: List[Dict[str, str]], 
    response_format: str = "json_object"
) -> Dict:
    """
    Call OpenRouter API directly using requests instead of the OpenAI client.
    
    Args:
        model: Model identifier
        messages: List of message objects
        response_format: Format for the response
        
    Returns:
        API response as a dictionary
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://continuous-learning-lab.local",
        "X-Title": "Continuous Learning Lab"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    
    # Add response_format for newer models that support it
    if response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()  # Raise exception for HTTP errors
    
    # Return the parsed JSON response
    return response.json()

def extract_content_from_response(response_json: Dict) -> str:
    """
    Extract content from the OpenRouter API response.
    
    Args:
        response_json: API response as a dictionary
        
    Returns:
        Content text
    """
    try:
        # Standard OpenAI format
        if "choices" in response_json and len(response_json["choices"]) > 0:
            choice = response_json["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        pass
    
    # Fallback: Log the full response and return an empty string
    log_event("unexpected_response_format", response=json.dumps(response_json)[:500])
    
    return ""

def extract_score_from_content(content: str) -> int:
    """
    Extract score from response content.
    
    Args:
        content: The content string, hopefully containing JSON
        
    Returns:
        Score value (0-10)
    """
    # Try parsing as JSON first
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "score" in data:
            score = int(data["score"])
            if 0 <= score <= 10:
                return score
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    # Try regex extraction
    import re
    score_match = re.search(r'"score"\s*:\s*(\d+)', content)
    if score_match:
        try:
            score = int(score_match.group(1))
            if 0 <= score <= 10:
                return score
        except (ValueError, IndexError):
            pass
    
    # Last resort: look for a standalone number between 0-10
    number_match = re.search(r'\b([0-9]|10)\b', content)
    if number_match:
        try:
            score = int(number_match.group(1))
            return score
        except (ValueError, IndexError):
            pass
    
    # Default fallback
    return 5

def grade_response(prompt: str, answer: str) -> Tuple[str, int, str]:
    """
    Have a teacher model grade the student's response.
    
    Args:
        prompt: Original prompt
        answer: Student's response
        
    Returns:
        Tuple of (model_name, score, raw_response)
    """
    system_prompt = (
        "You are an expert evaluator grading responses on a scale from 0 to 10. "
        "Evaluate the response to the given prompt based on accuracy, clarity, "
        "comprehensiveness, and reasoning quality.\n\n"
        "Return a JSON object with the following structure:\n"
        "{\n"
        '  "score": integer from 0-10,\n'
        '  "reasoning": "brief explanation of your grading"\n'
        "}"
    )
    
    user_prompt = f"PROMPT:\n{prompt}\n\nRESPONSE:\n{answer}"
    
    # Try teachers in random order to balance usage
    for model in random.sample(TEACHER_MODELS, len(TEACHER_MODELS)):
        try:
            # Clean up model name (remove leading backslashes)
            clean_model = model.strip('\\').strip()
            
            # Call API and handle response
            response_json = call_openrouter_api(
                model=clean_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract content from response
            content = extract_content_from_response(response_json)
            
            # Extract score from content
            score = extract_score_from_content(content)
            
            # Log the raw response for debugging
            log_event(
                "grading_response_received", 
                model=clean_model,
                content_length=len(content),
                score=score
            )
            
            return clean_model, score, content
            
        except Exception as e:
            log_event("grading_error", model=model, error=str(e))
    
    # If all models fail
    raise RuntimeError("All teacher models failed to grade the response")

def generate_improved_response(model: str, prompt: str) -> Tuple[str, str]:
    """
    Have a teacher model generate an improved response.
    
    Args:
        model: Model to use for correction
        prompt: Original prompt
        
    Returns:
        Tuple of (improved_response, raw_response)
    """
    system_prompt = (
        "You are a helpful, accurate assistant. Provide a comprehensive and "
        "well-reasoned response to the user's question or prompt. "
        "Follow these steps:\n"
        "1. Analyze what the prompt is asking for\n"
        "2. Think step-by-step through the solution or explanation\n"
        "3. Provide a clear, accurate, and helpful response\n"
        "4. When appropriate, explain your reasoning"
    )
    
    try:
        # Clean up model name
        clean_model = model.strip('\\').strip()
        
        # Call API and handle response
        response_json = call_openrouter_api(
            model=clean_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format="text"
        )
        
        # Extract content from response
        content = extract_content_from_response(response_json)
        
        # Log success
        log_event(
            "correction_response_received", 
            model=clean_model,
            content_length=len(content)
        )
        
        return content.strip(), content
    except Exception as e:
        log_event("correction_error", model=model, error=str(e))
        raise

def adapt_model() -> None:
    """
    Adapt the model based on collected examples.
    
    This function:
    1. Triggers LoRA updates when conditions are met
    2. Adjusts thresholds based on performance
    3. Updates metrics
    """
    global THRESHOLD, last_update_time, update_counter
    
    current_time = time.time()
    force_update = current_time - last_update_time > FORCE_UPDATE_SECS
    batch_ready = len(replay_buffer) >= BATCH_SIZE
    
    # Calculate acceptance rate
    acceptance_rate = 0
    mean_score = 0
    
    if score_window:
        acceptance_rate = sum(s >= THRESHOLD for s in score_window) / len(score_window)
        mean_score = statistics.mean(score_window)
    
    # Trigger model update when conditions are met
    if batch_ready or force_update:
        log_event(
            "adapting_model", 
            reason="batch_ready" if batch_ready else "force_update",
            examples=len(replay_buffer),
            acceptance_rate=acceptance_rate,
            mean_score=mean_score,
            threshold=THRESHOLD
        )
        
        # Train on collected examples
        examples_list = list(replay_buffer)
        if examples_list:
            train_lora(
                examples=examples_list,
                learning_rate=LEARNING_RATE,
                batch_size=TRAIN_BATCH_SIZE,
                epochs=TRAIN_EPOCHS
            )
        
        # Update metrics and reset state
        save_metric(update_counter, acceptance_rate, mean_score, THRESHOLD)
        update_counter += 1
        replay_buffer.clear()
        last_update_time = current_time
    
    # Adjust threshold based on performance
    if len(score_window) == WINDOW_SIZE and acceptance_rate < RATE_THRESHOLD and THRESHOLD > MIN_THRESHOLD:
        THRESHOLD -= 1
        log_event(
            "threshold_lowered", 
            new_threshold=THRESHOLD, 
            acceptance_rate=acceptance_rate
        )

# This task name matches what your system is expecting
@celery_app.task(name="student.grade")
def grade(items):
    """
    Evaluate a batch of student model responses and add to learning buffer if needed.
    
    Args:
        items: List of dictionaries containing the prompt, answer, qid, and topic
    """
    global score_window
    
    for item in items:
        prompt = item["prompt"]
        answer = item["answer"]
        qid = item["qid"]
        topic = item.get("topic", "general")
        
        try:
            # Grade the response
            teacher_model, score, raw_grade_response = grade_response(prompt, answer)
            
            # Log the score and save the teacher's grading response
            log_event(
                "response_graded", 
                qid=qid, 
                score=score, 
                model=teacher_model,
                topic=topic
            )
            
            save_teacher_response(
                qid=qid,
                model=teacher_model,
                prompt=prompt,
                response=raw_grade_response,
                score=score
            )
            
            # Add score to window
            score_window.append(score)
            
            # If score is good, add student's response to replay buffer
            if score >= THRESHOLD:
                replay_buffer.append((prompt, answer, "student"))
                log_event("example_added", qid=qid, source="student", score=score)
            else:
                # If score is below threshold, get improved response from teacher
                try:
                    improved_answer, raw_correction_response = generate_improved_response(
                        model=teacher_model,
                        prompt=prompt
                    )
                    
                    # Save the teacher's correction
                    save_teacher_response(
                        qid=qid,
                        model=teacher_model,
                        prompt=prompt,
                        response=raw_correction_response,
                        is_correction=True
                    )
                    
                    # Add the teacher's improved response to the replay buffer
                    replay_buffer.append((prompt, improved_answer, "teacher"))
                    log_event(
                        "example_added", 
                        qid=qid, 
                        source="teacher", 
                        score=score
                    )
                except Exception as e:
                    log_event("correction_failed", qid=qid, error=str(e))
            
            # Check if adaptation is needed
            adapt_model()
            
        except Exception as e:
            log_event("evaluation_failed", qid=qid, error=str(e))

# Also support the original task name format
@celery_app.task(name="tasks.evaluate_response")
def evaluate_response(prompt: str, answer: str, qid: str, topic: str) -> None:
    """
    Evaluate a single student model response.
    This is kept for compatibility with the API server.
    """
    grade([{"prompt": prompt, "answer": answer, "qid": qid, "topic": topic}])

if __name__ == "__main__":
    celery_app.start(["worker", "--loglevel=INFO", "--concurrency=1"])
