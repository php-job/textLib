import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.neighbors import NearestNeighbors 


class TxtDedubler:
    
    stopWords = [] 
    
    def addStopWord(self, word):
        self.stopWords.append(word)
        
    def tokenizer(self,X): 
        ps = PorterStemmer() 
        sentence = X
        words = word_tokenize(sentence)
        
        stemmer = SnowballStemmer("russian") 
 
        words=[stemmer.stem(word) for word in words]
        
        return words
        
    def bagOfWords(self,X): 
        cv = CountVectorizer(tokenizer=self.tokenizer)
        cv.fit(X.TEXT)  
        return cv.transform(X.TEXT)
        
    def train(self,X, tree_num=10): 
        
        if(isinstance(X, pd.core.frame.DataFrame)!=True):
            raise Exception('first argument must be pandas data frame')
        
        if(X.empty):
            raise Exception('The dataframe not has data')
        
        
        if('TEXT' not in X.columns or 'ID' not in X.columns):
            raise Exception('The dataframe mast have two required columns ID and TEXT')
        
        vector_view = self.bagOfWords(X)  
        self.wordsCount = vector_view.shape[1]
        window = self.wordsCount
           
        self.model = NearestNeighbors(algorithm='brute', metric='cosine').fit(vector_view) 
        
        return self.model, vector_view
      
    def predict(self,X, radius=0.1):
        model, vector_view = self.train(X)
        similar_ids = [] 
        similar_ids_len = [] 
        for i,row in X.iterrows():
            rng = model.radius_neighbors(vector_view[i], radius, return_distance=True)
            similar_ids.append(X[X.index.isin(rng[1][0])]['ID'].values)
            similar_ids_len.append(len(X[X.index.isin(rng[1][0])]['ID'].values))
        
        X['similar_ids'] = None
        X['similar_ids_len'] = None
        X['similar_ids'] = similar_ids
        X['similar_ids_len'] = similar_ids_len
        return X