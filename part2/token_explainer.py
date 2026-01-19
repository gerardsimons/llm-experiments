import os
import sys
import json
import hashlib
import argparse
import html
import re
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp
from scipy.stats import entropy
from diskcache import Cache
from tokenizers import Tokenizer

# Add the project root to the path to allow importing from `part2`
try:
    from part2.logprobs_cli import get_logprobs
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from part2.logprobs_cli import get_logprobs

# --- Caching & Logprob Setup ---
cache = Cache("./logprob_cache")

def _make_cache_key(prompt: str, logprob_kwargs: dict) -> str:
    payload = {"prompt": prompt, "kwargs": logprob_kwargs}
    dumped = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

def get_logprobs_cached(prompt: str, **logprob_kwargs):
    key = _make_cache_key(prompt, logprob_kwargs)
    if key in cache:
        return cache[key]
    res = get_logprobs(prompt=prompt, **logprob_kwargs)
    cache[key] = res
    return res

# --- Core Metrics ---

def log_margin(logprobs: Dict[str, float]) -> float:
    if len(logprobs) < 2: return float('inf')
    vals = sorted(logprobs.values(), reverse=True)
    return vals[0] - vals[1]

def predictive_entropy(logprobs: Dict[str, float]) -> float:
    if not logprobs: return 0.0
    lp = np.array(list(logprobs.values()))
    log_p = lp - logsumexp(lp)
    p = np.exp(log_p)
    return entropy(p)

# --- MASKING STRATEGIES ---

def get_masking_spans(text: str, strategy: str, tokenizer: Optional[Tokenizer] = None, phrase_size: int = 2) -> List[Dict[str, Any]]:
    spans = []
    if strategy == 'word':
        for match in re.finditer(r'\S+', text):
            spans.append({'span': match.span(), 'text': match.group()})
    
    elif strategy == 'phrase':
        words = list(re.finditer(r'\S+', text))
        if len(words) < phrase_size:
            return spans
        for i in range(len(words) - phrase_size + 1):
            start_char = words[i].start()
            end_char = words[i + phrase_size - 1].end()
            spans.append({
                'span': (start_char, end_char),
                'text': text[start_char:end_char]
            })

    elif strategy == 'subword':
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for 'subword' strategy.")
        encoding = tokenizer.encode(text)
        for i, (start, end) in enumerate(encoding.offsets):
            if encoding.tokens[i] in tokenizer.all_special_tokens:
                continue
            spans.append({
                'span': (start, end),
                'text': encoding.tokens[i]
            })
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}")
        
    return spans

# --- Masking Logic ---

def run_masking_experiment(
    prompt_template: str,
    txt: str,
    label: str,
    logprob_kwargs: Dict,
    strategy: str,
    relevant_labels: Optional[List[str]] = None,
    tokenizer: Optional[Tokenizer] = None,
    phrase_size: int = 2,
    mask_label: str = "[...]"
) -> pd.DataFrame:
    base_prompt = prompt_template.format(text=txt)
    base_logprobs = get_logprobs_cached(prompt=base_prompt, **logprob_kwargs).logprobs
    base_entropy = predictive_entropy(base_logprobs)
    
    spans_to_mask = get_masking_spans(txt, strategy, tokenizer, phrase_size)
    df_rows = []

    for i, span_info in enumerate(spans_to_mask):
        start, end = span_info['span']
        masked_txt = txt[:start] + mask_label + txt[end:]
        
        prompt = prompt_template.format(text=masked_txt)
        res = get_logprobs_cached(prompt=prompt, **logprob_kwargs)
        logprobs = res.logprobs

        row = {
            'token_index': i,
            'masked_word': span_info['text'],
            'entropy_change': predictive_entropy(logprobs) - base_entropy,
            'top_choice': res.response_text,
            'true_label': label,
        }

        for lbl in base_logprobs:
            if relevant_labels is not None and lbl not in relevant_labels:
                continue
            lp = logprobs.get(lbl)
            if lp is not None:
                row[f'delta_logprob_{lbl}'] = lp - base_logprobs[lbl]

        df_rows.append(row)

    return pd.DataFrame(df_rows)

def token_masking_analysis(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    pred_col: str,
    prompt_template: str,
    logprob_kwargs: Dict,
    strategy: str = 'word',
    relevant_labels: Optional[List[str]] = None,
    tokenizer: Optional[Tokenizer] = None,
    phrase_size: int = 2
) -> pd.DataFrame:
    mask_dfs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Analyzing with '{strategy}' strategy"):
        txt = row[text_col]
        label = row[label_col]

        mask_df = run_masking_experiment(
            prompt_template, txt, label, logprob_kwargs,
            strategy=strategy,
            relevant_labels=relevant_labels,
            tokenizer=tokenizer,
            phrase_size=phrase_size
        )
        
        if mask_df.empty:
            continue

        mask_df['original_text'] = txt
        mask_df['unmasked_pred'] = row.get(pred_col, 'N/A')

        competitor_options = mask_df['top_choice'].value_counts().index.difference([label])
        if not competitor_options.empty:
            competitor = competitor_options[0]
            mask_df['support_label'] = mask_df.get(f'delta_logprob_{label}')
            mask_df['suppress_competitor'] = -mask_df.get(f'delta_logprob_{competitor}', 0)
            mask_df['net_effect'] = mask_df['support_label'].fillna(0) + mask_df['suppress_competitor'].fillna(0)
        else:
            mask_df['support_label'] = mask_df.get(f'delta_logprob_{label}')
            mask_df['suppress_competitor'] = 0
            mask_df['net_effect'] = mask_df['support_label']

        mask_dfs.append(mask_df)
        
    if not mask_dfs: return pd.DataFrame()
    analysis_df = pd.concat(mask_dfs).reset_index(drop=True)
    analysis_df['prompt_id'] = analysis_df.groupby('original_text').ngroup()
    return analysis_df

# --- Visualization ---

def colorize_text_by_span(text: str, token_effects: Dict[int, float], spans: List[Dict[str, Any]]) -> str:
    segments = []
    last_idx = 0
    
    sorted_spans_with_effects = sorted(
        [(i, spans[i], token_effects.get(i)) for i in range(len(spans)) if token_effects.get(i) is not None],
        key=lambda x: x[1]['span'][0]
    )

    valid_effects = [v for v in token_effects.values() if v is not None and not np.isnan(v)]
    max_abs_effect = max(abs(v) for v in valid_effects) if valid_effects else 0

    def interpolate_color(effect):
        if effect is None or np.isnan(effect) or max_abs_effect == 0: return "#FFFFFF"
        normalized = effect / max_abs_effect
        r_bg, g_bg, b_bg = (200, 255, 200) if normalized > 0 else (255, 200, 200)
        alpha = abs(normalized)
        r, g, b = [int(c * alpha + 255 * (1 - alpha)) for c in (r_bg, g_bg, b_bg)]
        return f"#{r:02x}{g:02x}{b:02x}"

    for i, span_info, effect in sorted_spans_with_effects:
        start, end = span_info['span']
        if start > last_idx:
            segments.append(html.escape(text[last_idx:start]))
        
        color = interpolate_color(effect)
        segments.append(
            f'<span style="background-color: {color}; border-radius: 3px;">'
            f'{html.escape(text[start:end])}</span>'
        )
        last_idx = end

    if last_idx < len(text):
        segments.append(html.escape(text[last_idx:]))

    return "".join(segments)

def generate_html_report(analysis_df: pd.DataFrame, strategy: str, **kwargs) -> str:
    if analysis_df.empty: return "<h1>No analysis data to display.</h1>"
    
    report_parts = [
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Token Contribution Report</title>'
        '<style>body{font-family:sans-serif;line-height:1.6;margin:20px;background-color:#f4f4f4;color:#333;}'
        '.container{background-color:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:15px;margin-bottom:20px;}'
        'h1,h3{color:#0056b3;}</style></head><body><h1>Token Contribution Analysis</h1>'
    ]

    tokenizer = None
    if strategy == 'subword':
        tokenizer_path = kwargs.get('tokenizer_path', 'bert-base-uncased')
        tokenizer = Tokenizer.from_pretrained(tokenizer_path)

    for prompt_id, group in analysis_df.groupby('prompt_id'):
        group = group.sort_values('token_index')
        original_text = group['original_text'].iloc[0]
        true_label = group['true_label'].iloc[0]
        unmasked_pred = group['unmasked_pred'].iloc[0]

        spans = get_masking_spans(
            original_text, strategy, tokenizer=tokenizer, phrase_size=kwargs.get('phrase_size', 2)
        )
        
        token_effects = group.set_index('token_index')['net_effect'].to_dict()
        colorized_html = colorize_text_by_span(original_text, token_effects, spans)

        report_parts.append(f"<div class='container'><h3>Prompt ID: {prompt_id}</h3>")
        report_parts.append(f"<p><b>True Label:</b> <span style='color:green;'>{true_label}</span> | "
                            f"<b>Prediction:</b> <span style='color:red;'>{unmasked_pred}</span></p>")
        report_parts.append(f"<h4>Contribution Analysis (Strategy: {strategy}):</h4>"
                            f"<div style='padding:10px;border:1px solid #ccc;border-radius:5px;'>{colorized_html}</div></div>")

    report_parts.append("</body></html>")
    return "\n".join(report_parts)

def main():
    parser = argparse.ArgumentParser(description="Run token-level contribution analysis on a dataset.")
    parser.add_argument("input_csv", help="Path to input CSV file.")
    parser.add_argument("--text_col", required=True, help="Column with text.")
    parser.add_argument("--label_col", required=True, help="Column with true label.")
    parser.add_argument("--pred_col", required=True, help="Column with model predictions.")
    parser.add_argument("--prompt_template", required=True, help="Prompt template with a '{text}' placeholder.")
    parser.add_argument("--strategy", type=str, choices=['word', 'phrase', 'subword'], default='word', help="Masking strategy.")
    parser.add_argument("--phrase_size", type=int, default=2, help="Size of phrases for 'phrase' strategy.")
    parser.add_argument("--tokenizer_path", type=str, default="bert-base-uncased", help="Tokenizer path for 'subword' strategy.")
    parser.add_argument("--relevant_labels", type=str, help="Comma-separated labels to consider for delta_logprob columns.")
    parser.add_argument("--output_csv", default="analysis_results.csv", help="Path to save analysis CSV.")
    parser.add_argument("--output_html", default="analysis_report.html", help="Path to save HTML report.")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of rows to process.")
    parser.add_argument("--model_id", type=str, default='llama3:8b', help="Ollama model ID.")
    args = parser.parse_args()

    logprob_kwargs = {"provider": 'ollama', "model_id": args.model_id, "top_logprobs": 5, "temperature": 0.0, "invert_log": False}
    
    tokenizer = None
    if args.strategy == 'subword':
        try:
            print(f"Loading tokenizer '{args.tokenizer_path}'...")
            tokenizer = Tokenizer.from_pretrained(args.tokenizer_path)
        except Exception as e:
            print(f"Fatal: Could not load tokenizer. Subword strategy is not available. Error: {e}", file=sys.stderr)
            sys.exit(1)

    relevant_labels_list = [lbl.strip() for lbl in args.relevant_labels.split(',')] if args.relevant_labels else None

    try:
        df = pd.read_csv(args.input_csv)
        if args.limit > 0: df = df.head(args.limit)
    except FileNotFoundError:
        print(f"Error: Input CSV not found at {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    for col in [args.text_col, args.label_col, args.pred_col]:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in CSV.", file=sys.stderr)
            sys.exit(1)

    analysis_df = token_masking_analysis(
        df, args.text_col, args.label_col, args.pred_col, args.prompt_template, logprob_kwargs,
        strategy=args.strategy, relevant_labels=relevant_labels_list, tokenizer=tokenizer, phrase_size=args.phrase_size
    )

    if not analysis_df.empty:
        analysis_df.to_csv(args.output_csv, index=False)
        print(f"Analysis CSV saved to {args.output_csv}")

        html_report = generate_html_report(
            analysis_df, args.strategy, tokenizer_path=args.tokenizer_path, phrase_size=args.phrase_size
        )
        with open(args.output_html, 'w', encoding='utf-8') as f: f.write(html_report)
        print(f"HTML report saved to {args.output_html}")
    else:
        print("No analysis data generated.", file=sys.stderr)
    
    print("Done.")

if __name__ == "__main__":
    main()
