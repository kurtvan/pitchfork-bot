import pandas as pd
import numpy as np
import string, os 
import sqlite3

import nltk

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

        
# Read sqlite query results into a pandas DataFrame
con = sqlite3.connect("./database.sqlite")
df = pd.read_sql_query("SELECT * from content", con)
con.close()

all_reviews = list(df.content.values)

def clean_text(txt):
    txt = "".join(v for v in txt if (v not in string.punctuation or v in ['.'])).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 

corpus = [clean_text(x) for x in all_reviews]
corpus = " ".join(corpus)

tokens = nltk.word_tokenize(corpus)
text = nltk.Text(tokens)

print("generating...")
print(text.generate(500))