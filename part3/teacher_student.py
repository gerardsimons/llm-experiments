"""
Teacher-Student Learning with Logprob-based Signals as Features.

This script tests whether the log-probabilities from a "teacher" LLM can serve
as effective features for a simple "student" classifier to predict the
ground truth labels.

The experimental flow is as follows:
1. A teacher LLM processes a dataset of text and generates log-probabilities
   for each class (e.g., "Yes", "No").
2. These log-probabilities, along with derived metrics like the margin, are
   treated as a feature vector for each text sample.
3. A simple student model (e.g., Logistic Regression) is trained on these
   logprob features, with the goal of predicting the original ground truth labels.
4. The student is then evaluated on a test set to see how well the teacher's
   logprobs can be mapped to the true labels.
"""

import argparse
import logging
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from part2.logprobs import get_logprobs_cached
from part3.data_loader import load_spam_data
from part3.prompts import SPAM_ZERO_SHOT_PROMPT_V2

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Teacher: Feature Generation ---

def generate_logprob_features(texts, model_spec, labels):
    """
    Uses a teacher LLM to generate logprob-based features for a dataset.

    Args:
        texts (pd.Series): A series of texts to generate features for.
        model_spec (str): The teacher model identifier.
        labels (list): A list of possible string labels.

    Returns:
        pd.DataFrame: A DataFrame containing the generated features and the
                      teacher's argmax prediction, indexed by the original text index.
    """
    logging.info(f"Generating logprob features for {len(texts)} examples with teacher: {model_spec}...")
    
    provider, model_id = model_spec.split('/')
    feature_data = []

    for index, text in tqdm(texts.items(), desc="Generating Features", total=len(texts)):
        prompt = SPAM_ZERO_SHOT_PROMPT_V2.format(x=text, labels=labels)
        
        res = get_logprobs_cached(
            provider=provider,
            model_id=model_id,
            prompt=prompt,
            top_logprobs=len(labels) * 2,
            temperature=0,
        )
        
        logprobs_dict = {f"logprob_{lbl}": res.logprobs.get(lbl, -30.0) for lbl in labels}
        
        # Ensure we have valid logprobs before proceeding
        if all(val <= -30.0 for val in logprobs_dict.values()):
            logging.warning(f"Could not get valid logprobs for any label. Skipping index {index}.")
            continue

        # Calculate margin
        logprob_values = np.array(list(logprobs_dict.values()))
        order = np.argsort(logprob_values)[::-1]
        margin = logprob_values[order[0]] - logprob_values[order[1]]
        
        # Get argmax label
        teacher_prediction = labels[order[0]]
        
        # Combine all data for this sample
        sample_features = {
            "original_index": index,
            "teacher_prediction": teacher_prediction,
            "teacher_margin": margin,
            **logprobs_dict
        }
        feature_data.append(sample_features)
        
    if not feature_data:
        return pd.DataFrame()

    df = pd.DataFrame(feature_data)
    df = df.set_index("original_index")
    return df


# --- Student Model ---

def train_student_on_logprobs(logprob_features, true_labels, config):
    """
    Trains a Logistic Regression student on logprob features.
    """
    logging.info("Training student model on logprob features...")

    # Align features with labels
    aligned_df = true_labels.to_frame(name='true_label').join(logprob_features)
    aligned_df.dropna(inplace=True)
    
    feature_cols = [col for col in aligned_df.columns if 'logprob' in col or 'margin' in col]
    X_train = aligned_df[feature_cols]
    y_train = aligned_df['true_label']

    model = LogisticRegression(C=config['logreg']['C'], random_state=42)
    model.fit(X_train, y_train)
    
    return model, feature_cols


# --- Main Experiment Runner ---

def run_experiment(config):
    """
    Runs the full experiment: generate features, train student, evaluate.
    """
    logging.info(f"Running experiment with config: {config}")
    
    # 1. Load Data
    X_train_text, y_train_true, X_test_text, y_test_true = load_spam_data(
        train_size=config['data']['train_size'],
        test_size=config['data']['test_size']
    )
    labels = list(np.unique(y_train_true))
    
    # 2. Generate Logprob features for both train and test sets
    train_features_df = generate_logprob_features(X_train_text, config['teacher_model'], labels)
    test_features_df = generate_logprob_features(X_test_text, config['teacher_model'], labels)

    # 3. Train Student on Logprob Features to predict True Labels
    student_model, feature_cols = train_student_on_logprobs(train_features_df, y_train_true, config)

    # 4. Evaluate Student
    # Align test features with true test labels
    eval_df = y_test_true.to_frame(name='true_label').join(test_features_df)
    eval_df.dropna(inplace=True)

    if eval_df.empty:
        logging.error("No test samples could be processed by the teacher. Aborting evaluation.")
        return

    X_test_student = eval_df[feature_cols]
    y_test_true_aligned = eval_df['true_label']
    
    student_predictions = student_model.predict(X_test_student)
    student_accuracy = accuracy_score(y_test_true_aligned, student_predictions)

    # Also, check the teacher's own accuracy on this aligned test set
    teacher_accuracy = accuracy_score(y_test_true_aligned, eval_df['teacher_prediction'])

    # 5. Report Results
    logging.info("--- Experiment Results ---")
    logging.info(f"Teacher Model: {config['teacher_model']}")
    logging.info(f"Student Model: Logistic Regression on Logprob Features")
    logging.info("-" * 20)
    logging.info(f"Teacher's Raw Accuracy on Test Set: {teacher_accuracy:.4f}")
    logging.info(f"Student's Accuracy on Test Set: {student_accuracy:.4f}")
    logging.info("--------------------------")
    logging.info("Conclusion: This shows how well a simple linear model can learn to predict the true label using only the teacher's log-probabilities as features.")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run a teacher-student experiment using logprobs as features.")
    parser.add_argument(
        "--config",
        type=str,
        default="part3/config_gemini.yaml",
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    run_experiment(config)

if __name__ == "__main__":
    main()
