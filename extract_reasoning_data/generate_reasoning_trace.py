import os, json, time, argparse, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from typing import Optional, List, Dict, Any

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

# -----------------------------------------------------------------------
# Thread-local client => each thread gets its own connection object
_thread_local = threading.local()
def get_client() -> OpenAI:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(
            api_key="YOUR_DEEPSEEK_API_KEY",
            base_url=API_BASE,
        )
    return _thread_local.client

# -----------------------------------------------------------------------
def deepseek_reason(prompt: str) -> Dict[str, Any]:
    """
    Returns a dict with keys 'reasoning_trace' and 'output_answer'.
    """
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
            # 1) strip ``` fences if they slipped through
            if raw.startswith("```"):
                import re
                raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.S)
            # 2) load JSON
            payload = json.loads(raw)
            # 3) be forgiving about mis-quoted keys
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
            time.sleep(BACKOFF_SECS ** attempt)   # 2,4,8…
    return {"reasoning_trace": "", "output_answer": -1}

# ------------------------------------------------------------------------
def main(in_path: Path, out_path: Path, workers: int = 32):
    # ---------------------- load first 1000 rows -------------------------------
    with in_path.open() as f:
        rows: List[Dict[str, Any]] = [json.loads(line) for i, line in enumerate(f) if i < 1000]
    # Precompute prompts
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
         tqdm(total=len(rows), desc="DeepSeek-R1") as pbar:
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
    # --------------------- write jsonl output ---------------------------
    with out_path.open("w") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

# ------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",   dest="in_path", default="data/Gender_identity.jsonl")
    ap.add_argument("--out",  dest="out_path", default="data/Gender_identity_traces.jsonl")
    ap.add_argument("--workers", type=int, default=32, help="parallel DeepSeek calls (thread count)")
    args = ap.parse_args()
    main(Path(args.in_path), Path(args.out_path), args.workers)

