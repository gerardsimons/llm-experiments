import sys

import pandas as pd
from IPython.display import display, HTML # Note: IPython.display functions will only work in environments like Jupyter notebooks

from part2.token_explainer import (
    get_logprobs_cached,
    token_masking_analysis,
    generate_html_report,
    Tokenizer  # Import the tokenizer class
)

# ===================================================================
# 1. DEFINE YOUR INPUTS HERE
# ===================================================================
# Example 1: The original ISO language classification example
text_to_analyze = "os chefes de defesa da estónia, letónia, lituânia, alemanha, itália, espanha e eslováquia assinarão"
true_label = "pt"
prompt_template = """Given the text below, determine the single most appropriate ISO 639-1 language code.
Answer with the code only.

Text: {text}

Language code:"""
strategy = 'word' # Try 'word', 'phrase', or 'subword'
# You can also set a specific tokenizer path for 'subword' strategy
# tokenizer_path = "bert-base-uncased"
# And phrase_size for 'phrase' strategy
# phrase_size = 2


# ===================================================================
# 2. Define model parameters (no changes needed here usually)
# ===================================================================
logprob_kwargs = {
    "provider": 'ollama',
    "model_id": 'llama3:8b',
    "top_logprobs": 5,
    "temperature": 0.0,
    "invert_log": False
}


# ===================================================================
# 3. The rest of the script runs the analysis and displays the result
# ===================================================================
print("--- Starting Analysis ---")

# Get the initial prediction for the unmasked text
print("1. Getting initial prediction...")
initial_prompt = prompt_template.format(text=text_to_analyze)
initial_res = get_logprobs_cached(prompt=initial_prompt, **logprob_kwargs)
unmasked_pred = initial_res.response_text
print(f"   - Initial Prediction: '{unmasked_pred}' (True Label: '{true_label}')")

# Prepare a DataFrame for the analysis function
input_df = pd.DataFrame([{
    "text": text_to_analyze,
    "label": true_label,
    "preds": unmasked_pred
}])

# Load tokenizer only if needed for the 'subword' strategy
tokenizer = None
if strategy == 'subword':
    print("2. Loading 'subword' tokenizer...")
    # Using a generic tokenizer for demonstration.
    # For best results, this would match the model's actual tokenizer.
    # The default for the main CLI is "bert-base-uncased", let's use that here too.
    try:
        tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    except Exception as e:
        print(f"Error loading tokenizer: {e}. Subword strategy may not work without a valid tokenizer.", file=sys.stderr)
        print("You might need to download the tokenizer first, or specify a different path.", file=sys.stderr)
        tokenizer = None # Ensure tokenizer is None on failure
else:
    print("2. Skipping tokenizer loading.")

# Run the masking analysis
print(f"3. Running '{strategy}' masking analysis...")
# You can pass phrase_size=N here if using the 'phrase' strategy
analysis_df = token_masking_analysis(
    df=input_df,
    text_col='text',
    label_col='label',
    pred_col='preds',
    prompt_template=prompt_template,
    logprob_kwargs=logprob_kwargs,
    strategy=strategy,
    tokenizer=tokenizer,
    # relevant_labels=['pt', 'et', 'es'], # Example of using relevant_labels
    # phrase_size=phrase_size # Example of using phrase_size
)

# Generate and display the visual report
if not analysis_df.empty:
    print("4. Generating visual report...")
    html_report = generate_html_report(analysis_df, strategy=strategy) # Pass strategy for report details
    # display(HTML(html_report)) # This line will only work in IPython environments
    
    # For a standalone script, you might want to save the HTML to a file
    output_html_file = "experiment_report.html"
    with open(output_html_file, 'w', encoding='utf-8') as f:
        f.write(html_report)
    print(f"Report saved to {output_html_file}. Open it in a web browser to view.")

else:
    print("Could not generate analysis. The analysis dataframe was empty.")

print("--- Analysis Complete ---")
