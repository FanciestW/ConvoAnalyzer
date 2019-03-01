# ConvoAnalyzer [![Build Status](https://travis-ci.com/FanciestW/ConvoAnalyzer.svg?token=Mj9EgDohGpNEFd2iPtCp&branch=master)](https://travis-ci.com/FanciestW/ConvoAnalyzer) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/1827293c78b24e90af53a06ae410f98f)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=FanciestW/ConvoAnalyzer&amp;utm_campaign=Badge_Grade)
### By: William Lin

## Requirements:
- Python 3.6 or 3.7
- Pip3

## Usage:
```
usage: main.py [-h] [-i INPUT] [-o OUTPUT] [-d DIR] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Specifies the input .WAV file to be transcribed.
  -o OUTPUT, --output OUTPUT
                        Specifies the output file to write transcript to. If
                        none is specified, it will print to console.
  -d DIR, --dir DIR     Output directory for program output data. The default
                        is ./out/
  --debug               Flag to delete program output directory.
```

## How to Run:
1. Navigate into the project root directory.
2. Install all pip package dependencies by running:\
   ```pip install -r requirements.txt```
3. Run the program:\
   ```python3 main.py -i [input_wav_file]```