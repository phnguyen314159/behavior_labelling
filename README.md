note: remember to download nvidia cuda 12.1 and add the path to .env(create that file and add it to .gitignore)
## test
use this to run the test:
code>python -m pytest src/test.py    
## Setup (Windows CMD)

```bat
py -3.12 -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
python -m spacy download en_core_web_trf
