import subprocess

def run_script(script_name):
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

train = False
generation = True

if train:
    run_script('train.py')
if generation:
    run_script('generation.py')