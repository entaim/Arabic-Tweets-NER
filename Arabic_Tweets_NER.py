import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import warnings
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import re
import spacy
import string
import matplotlib.pyplot as plt
import plotly
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle 
nlp = spacy.blank('ar')
from spacy import displacy
warnings.filterwarnings('ignore')


st.title('Named Entity Recognition over Arabic Tweets')
st.write('_________________________________________________')
st.subheader('**This tool signify the goverment services mentioned in public tweets**')
#st.balloons()

consumerKey = 'confedential'
consumerSecret = 'confedential'
accessToken = 'confedential'
accessTokenSecret = 'confedential'

#-----------------------------Functions----------------------------------

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def lower_case_text(text):
    return text.lower()

def remove_stop_words(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop])


def lemmatize_words(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

remove_spaces = lambda x : re.sub('\\s+', ' ', x)

# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


#def convert_emojis(text):
    #for emot in UNICODE_EMOJI:
        #text = text.replace(emot, ' ')
    #return text

remove_double_quotes = lambda x : x.replace('"', '')
remove_single_quotes = lambda x : x.replace('\'', '')
trim = lambda x : x.strip()

other_chars = ['*', '#', '&x200B', '[', ']', '; ',' ;' "&nbsp", "“","“","”", "x200b"]


def remove_other_chars(x: str):
    for char in other_chars:
        x = x.replace(char, '')
    
    return x

token2idx = pickle.load(open("token2idx2-2.pkl", "rb"))
idx2tag = pickle.load(open("idx2tag2-2.pkl", "rb"))
idx2token = pickle.load(open("idx2token2-2.pkl", "rb"))


#token2idx = pickle.load(open("token2idx2_camel.pkl", "rb"))
#idx2tag = pickle.load(open("idx2tag2_camel.pkl", "rb"))
#idx2token = pickle.load(open("idx2token2_camel.pkl", "rb"))
stopper=15

    
def preprocess(tweet):
    tokens = tweet.split()
    data = pd.DataFrame({'Word': tokens})
    vocab = list(set(data['Word'].to_list()))
    data['Word_idx'] = data['Word'].map(token2idx)
    n_token = len(list(set(data['Word'].to_list())))
    data['Word_idx'].fillna(0, inplace=True)
    pad_tokens = pad_sequences([list(data.Word_idx)], maxlen=stopper, padding='post', value= n_token - 1)
  
    return pad_tokens[0]

#-----------------------------Logic--------------------------------------



#Create the authentication object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret) 
    
# Set the access token and access token secret
authenticate.set_access_token(accessToken, accessTokenSecret) 
    
# Creating the API object while passing in auth information
api = tweepy.API(authenticate, wait_on_rate_limit = True)

st.subheader('input your tweet query: ')
query = st.text_input(' ')

tweety = tweepy.Cursor(api.search_tweets,q=query,lang='ar').items(10)
tweets = [i.text for i in tweety]
#response = api.search(q = 'query', count = 100, geocode = '24.748538,46.695403,400mi')
#tweets = [response[i]._json['text'] for i in range(len(response))]



#model = keras.models.load_model('model2.h5')


model = keras.models.load_model('model4.h5')

tw = pd.DataFrame({'Tweet':tweets})



funcs = [
    remove_urls, 
    remove_punctuation,
    lower_case_text,
    remove_stop_words, 
    remove_emoji,
    remove_double_quotes, 
    remove_single_quotes,
    remove_other_chars,
    remove_spaces,
    trim]

st.write(tw)
for fun in funcs:
    tw['Tweet']= tw.Tweet.apply(fun)
    

disp=tw['Tweet'].apply(lambda x: x.split()).explode()
tw['Tweet']=tw['Tweet'].apply(preprocess)



pred = model.predict( np.asarray(tw['Tweet'].to_list()))

li = []
for i in pred: 
  for j in i:
    x = np.argmax(j)
    li.append(idx2tag[x])


disp.reset_index(drop=True, inplace=True)
disp= pd.concat([disp, pd.DataFrame({'label':li})], axis=1)

#st.write(disp)
if len(disp[disp['label']!='أخرى'])==0:
    st.warning("لم يتم العثور على خدمة حكومية")
else:
    st.write( disp[disp['label']!='أخرى'] )

#-------------------------------------------END---------------------------------------------------------
 


import plotly.express as px
x = pd.DataFrame(disp.label.value_counts())
fig = px.bar(x, x = x.index, y = x.label)
st.plotly_chart(fig)
