
import argparse
import logging
import os
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

# Project-specific imports
from part3.data_loader import load_data
from part3.evaluation import evaluate_and_save_results
from part3.classifiers.langchain_classifier import LangChainClassifier
from part3.classifiers.llamaindex_classifier import LlamaIndexClassifier
from part3.classifiers.scikit_llm_classifier import ScikitLLMClassifier
from part3.classifiers.instructor_classifier import InstructorClassifier

# --- Classifier Mapping ---
CLASSIFIERS = {
    "LangChain": LangChainClassifier,
    "LlamaIndex": LlamaIndexClassifier,
    "Scikit-LLM": ScikitLLMClassifier,
    "Instructor": InstructorClassifier,
}

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare LLM-based classification frameworks.")
    parser.add_argument(
        "--framework",
        type=str,
        choices=list(CLASSIFIERS.keys()) + ["all"],
        required=True,
        help="The framework to run."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ollama/llama3:8b",
        help="The model to use, prefixed with the provider (e.g., 'ollama/llama3', 'openai/gpt-4o-mini')."
    )
    parser.add_argument(
        "--scikit-llm-mode",
        type=str,
        default="zero-shot",
        choices=["zero-shot", "few-shot", "dynamic-shot"],
        help="The mode for the Scikit-LLM classifier."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"results/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Directory to save evaluation results."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level."
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="If set, prints the generated prompt for one sample before running predictions."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, runs the script with mock predictions to test the pipeline without API calls."
    )
    return parser.parse_args()

def main():
    """Main execution function."""
    args = get_args()
    console = Console()

    # --- Setup ---
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    os.makedirs(args.output_dir, exist_ok=True)
    
    console.print(f"[bold green]Starting LLM Classification Comparison[/bold green]")
    console.print(f"▸ Framework(s): [cyan]{args.framework}[/cyan]")
    console.print(f"▸ Model: [cyan]{args.model}[/cyan]")
    console.print(f"▸ Output Directory: [cyan]{args.output_dir}[/cyan]")
    if args.dry_run:
        console.print("[bold yellow]Running in DRY-RUN mode![/bold yellow]")

    # --- Data Loading ---
    X_train, y_train, X_test, y_test = load_data()

    # --- Framework Execution ---
    frameworks_to_run = CLASSIFIERS.keys() if args.framework == 'all' else [args.framework]

    for framework_name in frameworks_to_run:
        console.print(f"\n[bold]===== Running: {framework_name} =====[/bold]")

        try:
            classifier_class = CLASSIFIERS[framework_name]
            
            # Prepare classifier-specific arguments
            classifier_kwargs = {
                "model": args.model,
                "dry_run": args.dry_run,
                "show_prompt": args.show_prompt,
            }
            if framework_name == "Scikit-LLM":
                classifier_kwargs["mode"] = args.scikit_llm_mode

            # Initialize classifier
            classifier = classifier_class(**classifier_kwargs)
            
            # Skip incompatible classifiers
            if hasattr(classifier, 'is_compatible') and not classifier.is_compatible:
                continue

            # Fit and predict
            classifier.fit(X_train, y_train)
            
            if args.show_prompt:
                # Predict is responsible for populating the prompt
                predictions = classifier.predict(X_test)
                prompt = classifier.get_prompt_to_display()
                console.print(Panel(prompt, title="[yellow]Generated Prompt for First Item[/yellow]", border_style="yellow"))
            else:
                predictions = classifier.predict(X_test)

            # Evaluate
            evaluate_and_save_results(y_test, predictions, classifier.classes_, args.output_dir, framework_name)

        except Exception as e:
            console.print(f"[bold red]Error running {framework_name}: {e}[/bold red]")
            logging.error(f"Failed to run {framework_name}", exc_info=True)

    console.print("\n[bold green]Comparison run finished![/bold green]")

if __name__ == "__main__":
    main()
