"""
GPT4mole: Master Script for Training and Generation

This script provides a simple interface to run either training or generation
for the conditional molecular generation model.

Author: Wen Xing
License: MIT
"""

import subprocess


def run_script(script_name):
    """
    Execute a Python script and display its output.

    Args:
        script_name (str): Name of the Python script to execute
    """
    result = subprocess.run(['python', script_name],
                            capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)


# Configuration: Set which operations to perform
train = False       # Set to True to train a new model
generation = True   # Set to True to generate molecules

# Execute selected operations
if train:
    print("Starting model training...")
    run_script('train.py')

if generation:
    print("Starting molecule generation...")
    run_script('generation.py')

if not train and not generation:
    print("No operations selected. Please set train=True or generation=True.")
