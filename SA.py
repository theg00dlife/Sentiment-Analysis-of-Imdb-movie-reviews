
# coding: utf-8

# In[10]:

import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import TfidfVectorizer


# In[11]:

def review_to_words( raw_review , x):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    if x:
        meaningful_words = [w for w in words if not w in stops]   
        return( " ".join( meaningful_words ))#
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    
    return( " ".join( words ))


# In[12]:

train = pd.read_csv("labeledTrainData.tsv", header=0,                     delimiter="\t", quoting=3)

# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in xrange( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( review_to_words( train["review"][i] , 1) )


# In[13]:

def feature_extraction( method ):
    
    if method=="bagofwords":
        print "Creating the bag of words...\n"
        vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000) 
        train_data_features = vectorizer.fit_transform(clean_train_reviews) #returns Document-Term matrix
        train_data_features = train_data_features.toarray()
    
    if method=="bigramBOW":
        print "bigramBOW initiated .. \n"
        vectorizer =CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,stop_words = None, max_features = 5000, ngram_range=(2,2),token_pattern=r'\b\w+\b', min_df=1) 
        train_data_features = vectorizer.fit_transform(clean_train_reviews) #returns Document-Term matrix
        train_data_features = train_data_features.toarray()
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
    if method=="mixBOW":
        print "mixBOW initiated .. \n"
        vectorizer =CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,stop_words = None, max_features = 5000, ngram_range=(1,2),token_pattern=r'\b\w+\b', min_df=1) 
        train_data_features = vectorizer.fit_transform(clean_train_reviews) #returns Document-Term matrix
        train_data_features = train_data_features.toarray()
    
    if method=="tfidf":
        print "tfidf..\n"
        vectorizer =TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000, ngram_range=(1,1),token_pattern=r'\b\w+\b', min_df=1) 
        train_data_features = vectorizer.fit_transform(clean_train_reviews) #returns Document-Term matrix
        train_data_features = train_data_features.toarray()

    train_data_features=np.array(train_data_features)
    return train_data_features


# In[14]:

# Take a look at the words in the vocabulary
#vocab = vectorizer.get_feature_names()
#print vocab
# Sum up the counts of each vocabulary word
#dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
#for tag, count in zip(vocab, dist):
    #print count, tag


# In[15]:

acc = 0
#train_data_features=feature_extraction("bagofwords")
#train_data_features=feature_extraction("bigramBOW")
train_data_features=feature_extraction("mixBOW")

folds = KFold(train_data_features.shape[0],10)

i=0
for itrain,itest in folds:
    i+=1
    print i
    X_train = train_data_features[itrain]
    Y_train = train["sentiment"][itrain]
    X_test = train_data_features[itest]
    Y_test = train["sentiment"][itest]
    skm = LR(C=0.01,penalty='l1')
    skm.fit(X_train,Y_train)
    acc = acc + skm.score(X_test,Y_test)
    
print 'Total Average Accuracy :',(acc/10.0)*100


# In[16]:

#svm linear bow 83.884, 83.892
#svm linear bigram 83.8
#logreg mix 86.528. 87.436 (C=0.1)

