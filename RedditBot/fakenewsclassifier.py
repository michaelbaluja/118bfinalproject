from credentials import *
import pickle
import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

#get tfidf vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
#get mog model
with open('mog_loc.pkl', 'rb') as file:
    # Call load method to deserialze
    mog_model = pickle.load(file)

#get r/news and r/worldnews subreddits
subreddit = reddit.subreddit('News+WorldNews')
#get new posts from each
for sub in subreddit.stream.submissions():
    #prevent duplicate replies by checking if submission already saved
    if sub.saved:
        print("already replied: "+sub.title)
        continue
    #extract features from title
    vector_X = vectorizer.transform([sub.title]).toarray()
    #predict as real/fake news and probability of each
    prediction = mog_model.predict(vector_X)
    prob = mog_model.predict_proba(vector_X)
    #create output for reply
    predString = ''
    if prediction==0:
        predString='real news'
    else:
        predString='fake news'
    output = "BEEP BOOP \n"
    output+= "We have identified this article to be "+predString+"\n"
    output+= "Fake news probability: "+str(prob[0][1])+"\n"
    output+= "Real news probability: "+str(prob[0][0])+"\n"
    output+= "original headline classified from the title: \n"
    output+= sub.title
    print(output)
    #reply to submission with output
    #submission.reply(output) #uncomment this if actually going to reply
    #save submission to not reply again
    #sub.save() #uncomment this if actually going to reply