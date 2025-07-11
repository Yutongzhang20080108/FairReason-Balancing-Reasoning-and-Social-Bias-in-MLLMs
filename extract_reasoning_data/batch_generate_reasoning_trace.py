import os, json, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict, Any

MODEL_NAME   = "deepseek-reasoner"
API_BASE     = "https://api.deepseek.com"
MAX_RETRIES  = 3
BACKOFF_SECS = 2

PROMPT_TMPL = """You are a careful reasoner.

TASK
-----
Think step-by-step, then reply with **one** JSON object
containing two keys:

  "reasoning":  <your chain-of-thought in plain text>,
  "answer_index": <0|1|2>

Output *nothing* else – no markdown fences.

CONTEXT:
{context}

QUESTION:
{question}

OPTIONS:
0: {ans0}
1: {ans1}
2: {ans2}
"""

_thread_local = threading.local()
def get_client() -> OpenAI:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(
            api_key="YOUR_DEEPSEEK_API_KEY",
            base_url=API_BASE,
        )
    return _thread_local.client

def deepseek_reason(prompt: str) -> Dict[str, Any]:
    client = get_client()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = resp.choices[0].message.content
            if raw is not None:
                raw = raw.strip()
            else:
                raw = ""
            if raw.startswith("```"):
                import re
                raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.S)
            payload = json.loads(raw)
            def pick(d, canonical):
                for k in d:
                    if k.strip('"\'' ).strip() == canonical:
                        return d[k]
                raise KeyError(canonical)
            reasoning   = pick(payload, "reasoning")
            answer_idx  = int(pick(payload, "answer_index"))
            return {"reasoning_trace": reasoning, "output_answer": answer_idx}
        except Exception as exc:
            if attempt == MAX_RETRIES:
                return {"reasoning_trace": f"ERROR: {exc}", "output_answer": -1}
            time.sleep(BACKOFF_SECS ** attempt)
    return {"reasoning_trace": "", "output_answer": -1}

def process_file(input_path: Path, output_path: Path, workers: int = 32):
    with input_path.open() as f:
        rows: List[Dict[str, Any]] = [json.loads(line) for i, line in enumerate(f) if i < 5]
    prompts: List[str] = []
    for row in rows:
        try:
            prompt = PROMPT_TMPL.format(
                context = row["context"],
                question= row["question"],
                ans0    = row["ans0"],
                ans1    = row["ans1"],
                ans2    = row["ans2"],
            )
            prompts.append(prompt)
        except Exception as exc:
            prompts.append(f"ERROR: {exc}")
    results: List[Dict[str, Any]] = [{} for _ in range(len(rows))]
    with ThreadPoolExecutor(max_workers=workers) as pool, \
         tqdm(total=len(rows), desc=f"DeepSeek-R1: {input_path.name}") as pbar:
        futures = {
            pool.submit(deepseek_reason, prompt): idx
            for idx, prompt in enumerate(prompts)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result()
                rows[idx]["reasoning_trace"] = result.get("reasoning_trace", "")
                rows[idx]["output_answer"] = result.get("output_answer", -1)
            except Exception as exc:
                rows[idx]["reasoning_trace"] = f"ERROR: {exc}"
                rows[idx]["output_answer"] = -1
            pbar.update(1)
    with output_path.open("w") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    input_dir = Path("original_data")
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    files = [f for f in input_dir.iterdir() if f.suffix == ".jsonl"]
    for file in files:
        out_file = output_dir / f"{file.stem}_traces.jsonl"
        print(f"Processing {file} -> {out_file}")
        process_file(file, out_file, workers=32)

if __name__ == "__main__":
    main() 