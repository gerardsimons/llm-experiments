import numpy as np

def buggy_reorder_examples(examples, n_classes, n_examples):
    """The buggy implementation logic from skllm's BaseDynamicFewShotClassifier._reorder_examples.
    
    This function attempts to interleave examples from different classes.
    It assumes that for each of the `n_classes`, there are exactly `n_examples` available.
    """
    shuffled_list = []
    for i in range(n_examples):
        for cls_idx in range(n_classes):
            # This calculation (cls_idx * n_examples + i) generates indices
            # assuming a perfectly rectangular structure of examples:
            # [C0_E0, C0_E1, ..., C0_E(n-1), C1_E0, C1_E1, ..., C1_E(n-1), ...]
            # where C = class, E = example, n = n_examples
            shuffled_list.append(cls_idx * n_examples + i)
    
    print("\n--- BUGGY INTERLEAVE VERSION (from original library) ---")
    print(f"Total examples list length (what was actually gathered): {len(examples)}")
    print(f"Assumed total examples for generating indices (n_classes * n_examples): {n_classes * n_examples}")
    print(f"Calculated indices to access: {shuffled_list}")
    if shuffled_list:
        print(f"Maximum calculated index: {max(shuffled_list)}")

    # The error occurs here if any index in shuffled_list is >= len(examples)
    return [examples[i] for i in shuffled_list]

def fixed_reorder_examples(examples):
    """A robust implementation using random permutation (the fix applied).
    
    This function simply shuffles all available examples, which is safe
    regardless of how many examples each class provided.
    """
    shuffled_list = np.random.permutation(len(examples)).tolist()
    
    print("\n--- FIXED RANDOM SHUFFLE VERSION (applied fix) ---")
    print(f"Total examples list length: {len(examples)}")
    print(f"Calculated indices to access: {shuffled_list}")
    if shuffled_list:
        print(f"Maximum calculated index: {max(shuffled_list)}")

    return [examples[i] for i in shuffled_list]


if __name__ == "__main__":
    # --- Scenario that triggers the bug ---
    # Imagine a classifier is configured to seek 3 examples per class (n_examples=3)
    # and there are 2 distinct classes (n_classes=2).
    N_CLASSES_IN_DATASET = 2
    N_EXAMPLES_REQUESTED_PER_CLASS = 3

    # The `_get_prompt` method would collect examples for each class.
    # Let's say Class A has 3 examples, but Class B only has 1 example in the dataset.
    # The `examples` list (what `_get_prompt` actually passes to `_reorder_examples`)
    # would look like this:
    example_list_from_get_prompt = [
        "Class A - Example 1", # Index 0
        "Class A - Example 2", # Index 1
        "Class A - Example 3", # Index 2
        "Class B - Example 1", # Index 3 (Only one example for Class B)
    ]
    # Total length of `example_list_from_get_prompt` is 4.

    print("--- Demonstrating the original bug ---")
    try:
        buggy_reorder_examples(
            example_list_from_get_prompt, 
            N_CLASSES_IN_DATASET, 
            N_EXAMPLES_REQUESTED_PER_CLASS
        )
    except IndexError as e:
        print(f"\n>>> BUG CONFIRMED: Caught expected IndexError: {e}")
        print("   Explanation: The original logic tried to access index 4 or 5, but the list only has 4 items (indices 0-3).")
    
    print("\n" + "="*80 + "\n")

    print("--- Demonstrating the fix (random shuffle) ---")
    try:
        fixed_result = fixed_reorder_examples(example_list_from_get_prompt)
        print("\n>>> FIX CONFIRMED: The fixed version ran successfully, no IndexError.")
        print("   Reordered examples (order will vary due to random shuffle):")
        for ex in fixed_result:
            print(f"     - {ex}")
    except Exception as e:
        print(f"\n>>> ERROR IN FIXED VERSION (this should not happen): {e}")
