#!/usr/bin/env python3
"""
Usage:
  python okhra_cli.py --model-dir ./model --text "paste a paragraph here"
  python okhra_cli.py --model-dir ./model --input texts.jsonl
  python okhra_cli.py --model-dir ./model --input texts.txt
  cat file.txt | python okhra_cli.py --model-dir ./model --input -
"""

import argparse, json, sys, string, os
import numpy as np
import onnxruntime as ort


_PUNCT_STR = string.punctuation.replace("'", "") + (
    "\u2013\u2014\u2026\u00AB\u00BB"
    "\u2039\u203A\u201C\u201D\u2018\u2019\u201E\u201A"
    "\u2010\u2011\u2012\u2015\u201F\u201B"
)
_PUNCT_SET   = frozenset(_PUNCT_STR)
_TRANS_TABLE = str.maketrans({c: f"  {c}  " for c in _PUNCT_SET})


def syn_tokenize(text, select_words):
    text = text.lower().translate(_TRANS_TABLE)
    return " ".join(
        t if (t in _PUNCT_SET or t in select_words) else "_"
        for t in text.split()
    )


def text_to_indices(text, select_words, vocab_map, unk_idx=1):
    syn = syn_tokenize(text, select_words)
    return [vocab_map.get(t, unk_idx) for t in syn.split()]


def chunk(indices, max_len, overlap):
    step = max_len - overlap
    if len(indices) <= max_len:
        return [indices]
    chunks = [indices[i:i + max_len]
              for i in range(0, len(indices) - max_len + 1, step)]
    tail = indices[-max_len:]
    if chunks[-1] != tail:
        chunks.append(tail)
    return chunks



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict_single(text, session, vocab_map, select_words, config):
    indices = text_to_indices(text, select_words, vocab_map, config["unk_idx"])
    if not indices:
        return {"probability": 0.0, "num_chunks": 0}

    chunks = chunk(indices, config["max_len"], config["overlap"])
    B = len(chunks)
    L = config["max_len"]

    padded = np.zeros((B, L), dtype=np.int64)
    for i, c in enumerate(chunks):
        padded[i, :len(c)] = c

    logits = session.run(None, {"input_ids": padded})[0].flatten()
    probs = sigmoid(logits)

    return {
        "probability": float(np.mean(probs)),
        "num_chunks": B,
    }



def read_texts(path):
    """Read texts from .jsonl, .json, or plain .txt (one per line)."""
    if path == "-":
        lines = sys.stdin.read().strip().splitlines()
    else:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()

    if not lines:
        return []

    try:
        first = json.loads(lines[0])
        if isinstance(first, str):
            return [json.loads(l) for l in lines]
        if isinstance(first, dict) and "text" in first:
            return [json.loads(l)["text"] for l in lines]
        if isinstance(first, list):
            return first
    except (json.JSONDecodeError, KeyError):
        pass

    return lines



def main():
    ap = argparse.ArgumentParser(description="Okhra CLI (ONNX)")
    ap.add_argument("--model-dir", default="./model",
                    help="directory with model.onnx + config files")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--text", help="single text string to classify")
    grp.add_argument("--input", help="path to .jsonl/.txt file, or - for stdin")
    ap.add_argument("--fpr", type=float, default=0.01,
                    choices=[0.001, 0.005, 0.01, 0.02, 0.05],
                    help="target FPR (default: 0.01)")
    ap.add_argument("--output", help="output file (default: stdout)")
    args = ap.parse_args()

    d = args.model_dir
    config       = json.load(open(os.path.join(d, "model_config.json")))
    vocab_map    = json.load(open(os.path.join(d, "vocab.json")))
    select_words = set(json.load(open(os.path.join(d, "select_words.json"))))
    thresholds   = json.load(open(os.path.join(d, "thresholds.json")))

    threshold = thresholds[str(args.fpr)]["threshold"]

    # Create ONNX session
    providers = ort.get_available_providers()
    # Prefer GPU if available, fall back to CPU
    if "CUDAExecutionProvider" in providers:
        ep = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        ep = ["CPUExecutionProvider"]

    onnx_path = os.path.join(d, "model.onnx")
    if not os.path.isfile(onnx_path):
        print(f"Error: {onnx_path} not found.", file=sys.stderr)
        sys.exit(1)

    session = ort.InferenceSession(onnx_path, providers=ep)
    print(f"Loaded {onnx_path} (provider: {session.get_providers()[0]})",
          file=sys.stderr)

    texts = [args.text] if args.text else read_texts(args.input)
    out   = open(args.output, "w") if args.output else sys.stdout

    for text in texts:
        res = predict_single(text, session, vocab_map, select_words, config)
        label = "ai" if res["probability"] >= threshold else "human"
        record = {
            "prediction":   label,
            "probability":  round(res["probability"], 6),
            "threshold":    round(threshold, 6),
            "fpr":          args.fpr,
            "num_chunks":   res["num_chunks"],
            "text_preview": text[:120],
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.output:
        out.close()
        print(f"Wrote {len(texts)} predictions → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
