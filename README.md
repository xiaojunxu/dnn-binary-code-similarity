# Gemini Network
This repo provides an implementation of the Gemini network for binary code similarity detection in [this paper](https://arxiv.org/abs/1708.06525).

## Prepration and Data
Unzip the data by running:
```bash
unzip data.zip
```

The network is based on Tensorflow 1.4 in Python 2.7. You can install the dependency by running:
```bash
pip install -r requirements.txt
```

## Model Implementation
The model is implemented in `graphnnSiamese.py`. An example script to use the model is `exp.py`. Run the following code to execute the script:
```bash
python exp.py
```
or run `python exp.py -h` to check the optional arguments.
