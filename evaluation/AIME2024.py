import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
from io import BytesIO

class AIME2024Evaluator:
    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize AIME2024 evaluator with model"""
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_aime2024_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load AIME2024 dataset from JSON/JSONL file"""
        dataset = []
        path = Path(dataset_path)
        
        if path.suffix == ".json":
            with open(path, 'r') as f:
                data = json.load(f)
                dataset = data if isinstance(data, list) else [data]
        elif path.suffix == ".jsonl":
            with open(path, 'r') as f:
                for line in f:
                    if line.strip():
                        dataset.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return dataset

    def extract_answer(self, response: str) -> Optional[int]:
        """Extract numerical answer from model response"""
        # Look for patterns like "The answer is 123" or "Answer: 123"
        patterns = [
            r"(?:the\s+)?answer\s+is\s+(\d+)",
            r"answer:\s*(\d+)",
            r"final\s+answer:\s*(\d+)",
            r"therefore,?\s+(\d+)",
            r"thus,?\s+(\d+)",
            r"so,?\s+(\d+)",
            r"\b(\d+)\s*$"  # Number at the end
        ]
        
        response_lower = response.lower().strip()
        
        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        # Extract all numbers and take the last one as fallback
        numbers = re.findall(r'\b\d+\b', response)
        if numbers:
            try:
                return int(numbers[-1])
            except ValueError:
                pass
        
        return None

    def generate_response(self, problem: str, max_length: int = 2048, temperature: float = 0.1) -> str:
        """Generate model response for AIME problem"""
        prompt = f"""Solve the following AIME problem step by step. Show your reasoning and provide the final numerical answer.

Problem: {problem}

Solution:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        response = response[len(prompt):].strip()
        
        return response

    def evaluate_dataset(self, dataset: List[Dict[str, Any]], output_path: str = None) -> Dict[str, Any]:
        """Evaluate model on AIME2024 dataset"""
        results = []
        correct = 0
        total = 0
        
        for i, item in enumerate(dataset):
            problem = item.get("problem", item.get("question", ""))
            ground_truth = item.get("answer", item.get("label", None))
            
            if not problem:
                print(f"Skipping item {i}: No problem text found")
                continue
            
            if ground_truth is None:
                print(f"Skipping item {i}: No ground truth answer found")
                continue
            
            print(f"Processing problem {i+1}/{len(dataset)}")
            
            try:
                response = self.generate_response(problem)
                predicted_answer = self.extract_answer(response)
                
                is_correct = predicted_answer is not None and predicted_answer == int(ground_truth)
                if is_correct:
                    correct += 1
                
                total += 1
                
                result = {
                    "problem_id": i,
                    "problem": problem,
                    "ground_truth": int(ground_truth),
                    "model_response": response,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct
                }
                results.append(result)
                
                print(f"GT: {ground_truth}, Predicted: {predicted_answer}, Correct: {is_correct}")
                
            except Exception as e:
                print(f"Error processing problem {i}: {e}")
                result = {
                    "problem_id": i,
                    "problem": problem,
                    "ground_truth": int(ground_truth) if ground_truth is not None else None,
                    "model_response": "",
                    "predicted_answer": None,
                    "is_correct": False,
                    "error": str(e)
                }
                results.append(result)
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        evaluation_results = {
            "model_name": self.model.config.name_or_path,
            "dataset": "AIME2024",
            "total_problems": total,
            "correct_answers": correct,
            "accuracy": accuracy,
            "results": results
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")
        
        return evaluation_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on AIME2024 benchmark")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to AIME2024 dataset")
    parser.add_argument("--output_path", type=str, default="aime2024_results.json", help="Output file path")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}")
    evaluator = AIME2024Evaluator(args.model_path, args.device)
    
    print(f"Loading dataset from {args.dataset_path}")
    dataset = evaluator.load_aime2024_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} problems")
    
    print("Starting evaluation...")
    results = evaluator.evaluate_dataset(dataset, args.output_path)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {results['model_name']}")
    print(f"Dataset: {results['dataset']}")
    print(f"Total Problems: {results['total_problems']}")
    print(f"Correct Answers: {results['correct_answers']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()