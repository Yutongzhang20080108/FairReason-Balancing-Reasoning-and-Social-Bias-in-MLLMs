import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sympy
from sympy import sympify, latex
from sympy.parsing.latex import parse_latex

class MATH500Evaluator:
    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize MATH-500 evaluator with model"""
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_math500_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load MATH-500 dataset from JSON/JSONL file"""
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

    def normalize_answer(self, answer: str) -> str:
        """Normalize mathematical expressions for comparison"""
        if not answer:
            return ""
        
        # Remove LaTeX formatting
        answer = re.sub(r'\\text\{[^}]*\}', '', answer)
        answer = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', answer)
        answer = re.sub(r'\\[a-zA-Z]+', '', answer)
        
        # Remove common delimiters
        answer = answer.replace('$', '').replace('{', '').replace('}', '')
        answer = answer.replace('\\', '').replace(',', '')
        
        # Handle fractions
        fraction_pattern = r'(\d+)/(\d+)'
        def replace_fraction(match):
            num, den = match.groups()
            try:
                return str(float(num) / float(den))
            except:
                return match.group(0)
        answer = re.sub(fraction_pattern, replace_fraction, answer)
        
        # Extract number or expression
        answer = answer.strip().lower()
        
        # Try to evaluate as sympy expression
        try:
            expr = sympify(answer)
            return str(expr.evalf())
        except:
            pass
        
        # Extract final number if present
        numbers = re.findall(r'-?\d+\.?\d*', answer)
        if numbers:
            return numbers[-1]
        
        return answer.strip()

    def extract_answer(self, response: str) -> Optional[str]:
        """Extract answer from model response"""
        # Look for boxed answers first (common in MATH dataset)
        boxed_pattern = r'\\boxed\{([^}]*)\}'
        boxed_match = re.search(boxed_pattern, response)
        if boxed_match:
            return self.normalize_answer(boxed_match.group(1))
        
        # Look for answer patterns
        patterns = [
            r"(?:the\s+)?(?:final\s+)?answer\s+is\s+([^\n.]+)",
            r"answer:\s*([^\n.]+)",
            r"therefore,?\s+([^\n.]+)",
            r"thus,?\s+([^\n.]+)",
            r"so,?\s+([^\n.]+)",
            r"solution:\s*([^\n.]+)",
            r"result:\s*([^\n.]+)"
        ]
        
        response_lower = response.lower()
        
        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                answer = match.group(1).strip()
                normalized = self.normalize_answer(answer)
                if normalized:
                    return normalized
        
        # Look for mathematical expressions in the last few lines
        lines = response.strip().split('\n')
        for line in reversed(lines[-3:]):  # Check last 3 lines
            line = line.strip()
            if line and not line.startswith('therefore') and not line.startswith('thus'):
                normalized = self.normalize_answer(line)
                if normalized and re.match(r'^-?\d+\.?\d*$', normalized):
                    return normalized
        
        return None

    def generate_response(self, problem: str, max_length: int = 2048, temperature: float = 0.1) -> str:
        """Generate model response for MATH problem"""
        prompt = f"""Solve the following math problem step by step. Show your work clearly and provide the final answer.

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

    def check_answer_equivalence(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer is equivalent to ground truth"""
        if not predicted or not ground_truth:
            return False
        
        pred_norm = self.normalize_answer(predicted)
        gt_norm = self.normalize_answer(ground_truth)
        
        # Direct string comparison
        if pred_norm == gt_norm:
            return True
        
        # Numerical comparison with tolerance
        try:
            pred_float = float(pred_norm)
            gt_float = float(gt_norm)
            return abs(pred_float - gt_float) < 1e-6
        except:
            pass
        
        # Symbolic comparison using sympy
        try:
            pred_expr = sympify(pred_norm)
            gt_expr = sympify(gt_norm)
            return pred_expr.equals(gt_expr)
        except:
            pass
        
        return False

    def evaluate_dataset(self, dataset: List[Dict[str, Any]], output_path: str = None) -> Dict[str, Any]:
        """Evaluate model on MATH-500 dataset"""
        results = []
        correct = 0
        total = 0
        subject_stats = {}
        
        for i, item in enumerate(dataset):
            problem = item.get("problem", item.get("question", ""))
            ground_truth = item.get("solution", item.get("answer", ""))
            subject = item.get("subject", item.get("type", "unknown"))
            level = item.get("level", "unknown")
            
            if not problem:
                print(f"Skipping item {i}: No problem text found")
                continue
            
            if not ground_truth:
                print(f"Skipping item {i}: No ground truth answer found")
                continue
            
            print(f"Processing problem {i+1}/{len(dataset)} [{subject}] [Level {level}]")
            
            try:
                response = self.generate_response(problem)
                predicted_answer = self.extract_answer(response)
                
                is_correct = self.check_answer_equivalence(predicted_answer, ground_truth)
                if is_correct:
                    correct += 1
                
                total += 1
                
                # Track subject-wise statistics
                if subject not in subject_stats:
                    subject_stats[subject] = {"total": 0, "correct": 0}
                subject_stats[subject]["total"] += 1
                if is_correct:
                    subject_stats[subject]["correct"] += 1
                
                result = {
                    "problem_id": i,
                    "problem": problem,
                    "subject": subject,
                    "level": level,
                    "ground_truth": ground_truth,
                    "model_response": response,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct
                }
                results.append(result)
                
                print(f"GT: {ground_truth[:50]}..., Predicted: {predicted_answer}, Correct: {is_correct}")
                
            except Exception as e:
                print(f"Error processing problem {i}: {e}")
                result = {
                    "problem_id": i,
                    "problem": problem,
                    "subject": subject,
                    "level": level,
                    "ground_truth": ground_truth,
                    "model_response": "",
                    "predicted_answer": None,
                    "is_correct": False,
                    "error": str(e)
                }
                results.append(result)
                total += 1
        
        # Calculate subject-wise accuracies
        for subject in subject_stats:
            if subject_stats[subject]["total"] > 0:
                subject_stats[subject]["accuracy"] = subject_stats[subject]["correct"] / subject_stats[subject]["total"]
            else:
                subject_stats[subject]["accuracy"] = 0.0
        
        overall_accuracy = correct / total if total > 0 else 0
        
        evaluation_results = {
            "model_name": getattr(self.model.config, 'name_or_path', 'unknown'),
            "dataset": "MATH-500",
            "total_problems": total,
            "correct_answers": correct,
            "overall_accuracy": overall_accuracy,
            "subject_statistics": subject_stats,
            "results": results
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")
        
        return evaluation_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on MATH-500 benchmark")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to MATH-500 dataset")
    parser.add_argument("--output_path", type=str, default="math500_results.json", help="Output file path")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}")
    evaluator = MATH500Evaluator(args.model_path, args.device)
    
    print(f"Loading dataset from {args.dataset_path}")
    dataset = evaluator.load_math500_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} problems")
    
    print("Starting evaluation...")
    results = evaluator.evaluate_dataset(dataset, args.output_path)
    
    print("\n" + "="*60)
    print("MATH-500 EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {results['model_name']}")
    print(f"Dataset: {results['dataset']}")
    print(f"Total Problems: {results['total_problems']}")
    print(f"Correct Answers: {results['correct_answers']}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print("\nSubject-wise Accuracy:")
    print("-" * 40)
    
    for subject, stats in results['subject_statistics'].items():
        print(f"{subject}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.4f})")
    
    print("="*60)

if __name__ == "__main__":
    main()