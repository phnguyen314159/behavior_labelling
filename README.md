note: remember to download nvidia cuda 12.1 and add the path to .env(create that file and add it to .gitignore)
and use this to get vs builttool to get desktop dev with c++, make sure you have msvc and w11 sdk: https://aka.ms/vs/17/release/vs_buildtools.exe
for.env, remember to replace \ with / or //
## test
use this to run the test:
code>python -m pytest src/test.py    
## Setup (Windows CMD)

```bat
py -3.12 -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
python -m spacy download en_core_web_trf
