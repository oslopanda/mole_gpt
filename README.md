# mole_GPT

mole_GPT is a GPT like transformer based conditional molecule generator and based on that a high drug-likeness (QED) dataset was generated.

## Initialise

clone this repo:

```bash
git clone https://github.com/oslopanda/mole_gpt
```
download the training dataset
....

download the pre-trained .pth file, if you don't want to train the model again.
....

## Usage
In the master.py, chose if you want to train a model or make generations.
```python
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
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)