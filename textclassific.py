# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk

df = pd.read_csv('C:/Users/raiajay/Desktop/kartikey/Womens Clothing E-Commerce Reviews.csv')
df.head()
df.tail()
df.isna().sum()
df.fillna("",inplace=True)
df.head()
#img_coloring = np.array(Image.open(path.join("../input/clothing2/","clothing2.jpg")))
x = df['Review Text'].values
wc = WordCloud(background_color = 'white',max_words = 2000, max_font_size = 40,stopwords = STOPWORDS,random_state = 42,width = 1600 ,height = 800).generate(" ".join(x))