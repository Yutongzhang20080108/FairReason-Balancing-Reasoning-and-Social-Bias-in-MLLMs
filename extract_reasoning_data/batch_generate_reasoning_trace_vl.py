import json
import base64
import requests
from pathlib import Path
from openai import OpenAI
import time
from typing import Dict, Any

def encode_image(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_reasoning_trace(client: OpenAI, image_path: str, question: str, model: str = "gpt-4-vision-preview") -> str:
    """Generate reasoning trace for vision question using OpenAI API"""
    try:
        base64_image = encode_image(image_path)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Please provide step-by-step reasoning for the following question about the image:\n\nQuestion: {question}\n\nProvide detailed reasoning steps leading to your final answer."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating reasoning trace: {e}")
        return ""

def process_dataset(input_path: Path, output_path: Path, api_key: str, 
                   image_field: str = "image_path", question_field: str = "question",
                   model: str = "gpt-4-vision-preview", batch_size: int = 10):
    """Process dataset to generate reasoning traces"""
    client = OpenAI(api_key=api_key)
    
    processed_count = 0
    error_count = 0
    
    with input_path.open() as fin, output_path.open("w") as fout:
        for line_num, line in enumerate(fin):
            if not line.strip():
                continue
                
            try:
                row = json.loads(line)
                image_path = row.get(image_field)
                question = row.get(question_field)
                
                if not image_path or not question:
                    print(f"Line {line_num + 1}: Missing image_path or question")
                    error_count += 1
                    continue
                
                if not Path(image_path).exists():
                    print(f"Line {line_num + 1}: Image file not found: {image_path}")
                    error_count += 1
                    continue
                
                print(f"Processing line {line_num + 1}: {image_path}")
                
                reasoning_trace = generate_reasoning_trace(client, image_path, question, model)
                
                if reasoning_trace:
                    row["reasoning_trace"] = reasoning_trace
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    processed_count += 1
                else:
                    error_count += 1
                
                # Rate limiting
                if processed_count % batch_size == 0:
                    print(f"Processed {processed_count} items, sleeping for 1 second...")
                    time.sleep(1)
                    
            except json.JSONDecodeError:
                print(f"Line {line_num + 1}: Invalid JSON")
                error_count += 1
            except Exception as e:
                print(f"Line {line_num + 1}: Error processing: {e}")
                error_count += 1
    
    print(f"Processing complete. Processed: {processed_count}, Errors: {error_count}")

def main():
    import os
    
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()
    
    # Process all JSONL files in data directory
    input_files = [f for f in data_dir.iterdir() if f.suffix == ".jsonl" and not f.name.endswith("_traces.jsonl")]
    
    for input_file in input_files:
        output_file = data_dir / f"{input_file.stem}_traces.jsonl"
        print(f"Processing {input_file} -> {output_file}")
        
        process_dataset(
            input_path=input_file,
            output_path=output_file,
            api_key=api_key,
            model="gpt-4-vision-preview",  # or "gpt-4o" for newer model
            batch_size=10
        )

if __name__ == "__main__":
    main()