#!/usr/bin/env python3
"""
Usage:
  python okhra_cli.py --model-dir ./model --text "paste a paragraph here"
  python okhra_cli.py --model-dir ./model --input texts.jsonl
  python okhra_cli.py --model-dir ./model --input texts.txt
  cat file.txt | python okhra_cli.py --model-dir ./model
"""

import argparse, json, sys, string, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Okhra(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=256,
                 hidden_dim=256, padding_idx=0, max_len=150,
                 convs=(2, 3, 4, 5, 7)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in convs])
        self.bns   = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in convs])
        self.dropout = nn.Dropout(0.15)
        fc1_in = num_filters * len(convs) * 2 + 1
        self.fc1   = nn.Linear(fc1_in, hidden_dim)
        self.bn_fc = nn.BatchNorm1d(hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths=None):
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pos = pos.clamp(max=self.pos_embedding.num_embeddings - 1)
        x = self.embedding(x) + self.pos_embedding(pos)
        x = x.permute(0, 2, 1)
        pooled = []
        for conv, bn in zip(self.convs, self.bns):
            h = F.gelu(bn(conv(x)))
            pooled.extend([
                F.adaptive_max_pool1d(h, 1).squeeze(-1),
                F.adaptive_avg_pool1d(h, 1).squeeze(-1),
            ])
        feat = torch.cat(pooled, dim=1)
        if lengths is not None:
            feat = torch.cat([feat, (lengths / T).unsqueeze(1)], dim=1)
        feat = self.dropout(feat)
        feat = F.gelu(self.bn_fc(self.fc1(feat)))
        feat = self.dropout(feat)
        return self.fc2(feat).squeeze(-1)


_PUNCT_STR = string.punctuation.replace("'", "") + "\u2013\u2014\u2026\u00AB\u00BB" \
             "\u2039\u203A\u201C\u201D\u2018\u2019\u201E\u201A\u2010\u2011\u2012" \
             "\u2015\u201F\u201B"
_PUNCT_SET   = frozenset(_PUNCT_STR)
_TRANS_TABLE = str.maketrans({c: f"  {c}  " for c in _PUNCT_SET})


def syn_tokenize(text, select_words):
    text = text.lower().translate(_TRANS_TABLE)
    return " ".join(t if (t in _PUNCT_SET or t in select_words) else "_"
                    for t in text.split())


def text_to_indices(text, select_words, vocab_map, unk_idx=1):
    syn = syn_tokenize(text, select_words)
    return [vocab_map.get(t, unk_idx) for t in syn.split()]


def chunk(indices, max_len, overlap):
    step = max_len - overlap
    if len(indices) <= max_len:
        return [indices]
    chunks = [indices[i:i+max_len]
              for i in range(0, len(indices) - max_len + 1, step)]
    tail = indices[-max_len:]
    if chunks[-1] != tail:
        chunks.append(tail)
    return chunks


@torch.no_grad()
def predict_single(text, model, vocab_map, select_words, config, device):
    indices = text_to_indices(text, select_words, vocab_map, config["unk_idx"])
    if not indices:
        return {"probability": 0.0, "chunks": []}
    chunks = chunk(indices, config["max_len"], config["overlap"])
    lengths = torch.tensor([len(c) for c in chunks], dtype=torch.float32, device=device)
    padded  = torch.zeros(len(chunks), config["max_len"], dtype=torch.long, device=device)
    for i, c in enumerate(chunks):
        padded[i, :len(c)] = torch.tensor(c, dtype=torch.long)
    logits = model(padded, lengths)
    probs  = torch.sigmoid(logits).cpu().tolist()
    return {"probability": float(np.mean(probs)), "chunks": probs}


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
        if isinstance(first, str):                       # JSON array was split
            return [json.loads(l) for l in lines]
        if isinstance(first, dict) and "text" in first:  # JSONL
            return [json.loads(l)["text"] for l in lines]
        if isinstance(first, list):                      # single JSON array
            return first
    except (json.JSONDecodeError, KeyError):
        pass

    return lines                                          # plain text



def main():
    ap = argparse.ArgumentParser(description="Okhra CLI")
    ap.add_argument("--model-dir", required=True, help="directory with model files")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--text",  help="single text string to classify")
    grp.add_argument("--input", help="path to .jsonl/.txt file, or - for stdin")
    ap.add_argument("--fpr",    type=float, default=0.01,
                    choices=[0.001, 0.005, 0.01, 0.02, 0.05],
                    help="target FPR (default: 0.01)")
    ap.add_argument("--output", help="output file (default: stdout)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()

    d = args.model_dir
    config       = json.load(open(os.path.join(d, "model_config.json")))
    vocab_map    = json.load(open(os.path.join(d, "vocab.json")))
    select_words = set(json.load(open(os.path.join(d, "select_words.json"))))
    thresholds   = json.load(open(os.path.join(d, "thresholds.json")))

    threshold = thresholds[str(args.fpr)]["threshold"]

    device = torch.device(args.device)
    model = Okhra(config["vocab_size"],
                  embed_dim=config["embed_dim"],
                  num_filters=config["num_filters"],
                  hidden_dim=config["hidden_dim"],
                  padding_idx=config["pad_idx"],
                  max_len=config["max_len"],
                  convs=tuple(config["conv_kernels"])).to(device)
    model.load_state_dict(torch.load(
        os.path.join(d, "best_model.pt"), map_location=device, weights_only=True))
    model.eval()

    texts = [args.text] if args.text else read_texts(args.input)
    out   = open(args.output, "w") if args.output else sys.stdout

    for text in texts:
        res = predict_single(text, model, vocab_map, select_words, config, device)
        label = "ai" if res["probability"] >= threshold else "human"
        record = {
            "prediction":  label,
            "probability": round(res["probability"], 6),
            "threshold":   round(threshold, 6),
            "fpr":         args.fpr,
            "num_chunks":  len(res["chunks"]),
            "text_preview": text[:120],
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.output:
        out.close()
        print(f"Wrote {len(texts)} predictions → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
