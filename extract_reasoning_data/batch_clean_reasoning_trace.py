import json
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
            if str(row.get(answer_field)) == str(row.get(label_field)):
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept_lines += 1
            else:
                deleted_lines += 1
    print(f"{input_path.name}: Original lines: {original_lines}, Deleted lines: {deleted_lines}, Left lines: {kept_lines}")

def main():
    data_dir = Path("data")
    files = [f for f in data_dir.iterdir() if f.name.endswith("_traces.jsonl")]
    for file in files:
        out_file = data_dir / f"{file.stem}_cleaned.jsonl"
        clean_reasoning_trace(file, out_file)

if __name__ == "__main__":
    main() 