import json
import argparse
from pathlib import Path

def clean_reasoning_trace(input_path: Path, output_path: Path, label_field: str = "label", answer_field: str = "output_answer"):
    original_lines = 0
    deleted_lines = 0
    kept_lines = 0
    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            original_lines += 1
            row = json.loads(line)
            # Compare output_answer to label
            if str(row.get(answer_field)) == str(row.get(label_field)):
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept_lines += 1
            else:
                deleted_lines += 1
    print(f"Original lines: {original_lines}")
    print(f"Deleted lines: {deleted_lines}")
    print(f"Left lines: {kept_lines}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", default="data/Gender_identity_traces.jsonl")
    parser.add_argument("--out", dest="output_path", default="data/Gender_identity_traces_cleaned.jsonl")
    parser.add_argument("--label_field", default="label")
    parser.add_argument("--answer_field", default="output_answer")
    args = parser.parse_args()
    clean_reasoning_trace(Path(args.input_path), Path(args.output_path), args.label_field, args.answer_field) 