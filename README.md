## Setup (Windows CMD)

```bat
py -3.12 -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
python -m spacy download en_core_web_trf
