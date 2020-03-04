# from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re

REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = [line.rstrip('\n') for line in open('library/nltk_data/corpora/stopwords/english', 'r')]

def clean_text(text):
    # text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    # text = text.encode('ascii', errors='ignore').decode("utf-8")
    # text = text.encode('ascii', errors='ignore').decode("unicode_escape")
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub(' ', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text