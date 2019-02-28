# ConvoAnalyzer [![Build Status](https://travis-ci.com/FanciestW/ConvoAnalyzer.svg?token=Mj9EgDohGpNEFd2iPtCp&branch=master)](https://travis-ci.com/FanciestW/ConvoAnalyzer)
### By: William Lin

## Requirements:
- Python 3.6 or 3.7
- Pip3

## Usage:
```
usage: main.py [-h] [-i INPUT] [-o OUTPUT] [-d]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Specifies the input .WAV file to be transcribed.
  -o OUTPUT, --output OUTPUT
                        Specifies the output directory name for program
                        outputs. If none is specified, the default is ./out/
  -d, --delete          Flag to delete program output directory.
```

## How to Run:
1. Navigate into the project root directory.
2. Install all pip package dependencies by running:\
   ```pip install -r requirements.txt```
3. Run the program:\
   ```python3 main.py -i [input_wav_file]```