from __future__ import absolute_import

import numpy as np
import pandas as pd
from cuzco.language import rake
from bs4 import BeautifulSoup as bs
from itertools import islice
import nltk
import re
from nltk.corpus import stopwords
from os import listdir
from os import getcwd
from os.path import isfile, join
from . import language
from .language import get_articles


### extract keywords from first 2 sentences of article 
### works best for news-style articles, less well for 'stories'
def two_lines(t):
    return '.'.join(t.split('.')[:3])  ## headline + 2 sentences

def find_keywords(ls):
    stoplist = getcwd() + '/cuzco/language/SmartStoplist.txt'
    
    rake_object = rake.Rake(stoplist, 3, 4, 1)
    keywords = rake_object.run(ls)
    return keywords

def find_names(lt, slist):
    tokens = nltk.word_tokenize(lt)
    words = [words for words in tokens if ((words.lower() not in slist))]
    names = [w for w in words if re.search('^[A-Z][a-z]+$', w)]
    return names

def keywords(text_from_dataframe):
    
    stoplist = getcwd() + '/cuzco/language/SmartStoplist.txt'
    
    text = text_from_dataframe
    
    text = text.replace('.', '. ')
    lines = two_lines(text)
    keys = find_keywords(lines)
    keys = [k for k, value in keys if value >= 4.0]
    keys_all = find_keywords(text)
    keys_all = [k for k, value in keys_all if value >= 4.0]
    people = find_names(lines, stoplist)
    people = np.unique(people)
    people_all = find_names(text, stoplist)
    people_all = np.unique(people_all)
    
    output_keywords = dict()
    output_keywords['keywords_2lines'] = keys
    output_keywords['keywords_alltext'] = keys_all
    output_keywords['people_places_2lines'] = people
    output_keywords['people_places_all'] = people_all
    
    
    return output_keywords


    
    
def article_keywords(paper, first, last):
    
    views_df = get_articles.get_articles_contents(paper, first, last)
    
    #views_df['article_keywords'] = ''
    list_keywords = []
    for [_, article] in views_df.iterrows():
        
        text = str(article['headline']) +'. ' + str(article['content'])    
        
        #article['article_keywords'] = keywords(text)
        
        list_keywords += [keywords(text)]
    
    views_df['article_keywords'] = list_keywords
    
    return views_df
    
    
    #print (text_data_frame.head())
    
    #article_keywords = pd.DataFrame(0, index = np.arange(len(texts)))
    
    
    
    


    