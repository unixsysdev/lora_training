"""
Batch testing utility for the Continuous Learning Lab.
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Configuration
PORT = int(os.getenv("API_PORT", "8000"))
API_URL = f"http://localhost:{PORT}"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run batch tests on the model")
    
    parser.add_argument(
        "file", 
        type=str, 
        help="File containing prompts (one per line)"
    )
    
    parser.add_argument(
        "--topic", 
        type=str, 
        default="batch_test",
        help="Topic label for the batch (default: batch_test)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="results.jsonl",
        help="Output file for results (default: results.jsonl)"
    )
    
    parser.add_argument(
        "--delay", 
        type=float, 
        default=0.5,
        help="Delay between requests in seconds (default: 0.5)"
    )
    
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=512,
        help="Maximum tokens to generate per response (default: 512)"
    )
    
    parser.add_argument(
        "--temp", 
        type=float, 
        default=0.7,
        help="Temperature for generation (default: 0.7)"
    )
    
    return parser.parse_args()

def load_prompts(file_path):
    """Load prompts from file."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found - {file_path}")
        sys.exit(1)
    
    with open(path, 'r', encoding='utf-8') as f:
        # Strip whitespace and filter out empty lines
        return [line.strip() for line in f if line.strip()]

def main():
    """Run the batch test."""
    args = parse_args()
    prompts = load_prompts(args.file)
    
    print(f"Loaded {len(prompts)} prompts from {args.file}")
    print(f"Topic: {args.topic}")
    print(f"Output file: {args.output}")
    print(f"Request delay: {args.delay} seconds")
    
    results = []
    failed = 0
    
    # Create a progress bar
    with tqdm(total=len(prompts), desc="Testing") as pbar:
        for i, prompt in enumerate(prompts):
            try:
                # Call the API
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "prompt": prompt,
                        "topic": args.topic,
                        "max_new_tokens": args.max_tokens,
                        "temperature": args.temp
                    },
                    timeout=60
                )
                
                # Process response
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        "prompt": prompt,
                        "qid": result["qid"],
                        "answer": result["answer"],
                        "tokens": result.get("tokens", -1),
                        "timestamp": time.time()
                    })
                    
                    # Write results as we go to avoid losing data
                    with open(args.output, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(results[-1], ensure_ascii=False) + '\n')
                else:
                    failed += 1
                    print(f"\nError ({response.status_code}): {response.text}")
            
            except Exception as e:
                failed += 1
                print(f"\nException: {str(e)}")
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({"failed": failed})
            
            # Delay between requests
            if i < len(prompts) - 1:
                time.sleep(args.delay)
    
    # Print summary
    print(f"\nBatch test completed:")
    print(f"  Total prompts: {len(prompts)}")
    print(f"  Successful: {len(results)}")
    print(f"  Failed: {failed}")
    print(f"  Results saved to: {args.output}")

if __name__ == "__main__":
    main()
