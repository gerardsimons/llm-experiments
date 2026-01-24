
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.figure_factory as ff

def evaluate_and_save_results(y_test, y_pred, labels, output_dir, framework_name):
    """
    Calculates evaluation metrics and saves results to the output directory.

    Args:
        y_test: True labels.
        y_pred: Predicted labels.
        labels (list): A list of all unique class labels.
        output_dir (str): The directory to save the output files.
        framework_name (str): The name of the framework being evaluated.
    """
    print(f"\n--- Evaluating {framework_name} ---")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # 2. Classification Report
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    print("Classification Report:")
    print(report)

    # 3. Save raw results to CSV
    results_df = pd.DataFrame({
        'text': y_test.index,  # Assuming index is the text, need to fix this
        'true_label': y_test.values,
        'predicted_label': y_pred
    })
    csv_path = os.path.join(output_dir, f"{framework_name}_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved raw results to {csv_path}")

    # 4. Confusion Matrix (static PNG)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(f'Confusion Matrix - {framework_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_path = os.path.join(output_dir, f"{framework_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # 5. Interactive Confusion Matrix (HTML)
    try:
        z = cm.tolist()
        x = labels
        y = labels
        # invert z
        z_text = [[str(y) for y in x] for x in z]
        
        fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
        fig.update_layout(title_text=f'<i><b>Confusion Matrix - {framework_name} (Interactive)</b></i>')
        fig.add_annotation(dict(font=dict(color="black",size=14),
                                x=0.5,
                                y=-0.15,
                                showarrow=False,
                                text="Predicted label",
                                xref="paper",
                                yref="paper"))
        fig.add_annotation(dict(font=dict(color="black",size=14),
                                x=-0.1,
                                y=0.5,
                                showarrow=False,
                                text="Actual label",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))
        fig.update_layout(margin=dict(t=50, l=200))
        fig['data'][0]['showscale'] = True
        
        interactive_cm_path = os.path.join(output_dir, f"{framework_name}_confusion_matrix_interactive.html")
        fig.write_html(interactive_cm_path)
        print(f"Saved interactive confusion matrix to {interactive_cm_path}")
    except Exception as e:
        print(f"Could not create interactive confusion matrix: {e}")
