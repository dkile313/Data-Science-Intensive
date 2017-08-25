import numpy as np
# from memory_profiler import profile
import pandas as pd
import time
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import string
from timeit import default_timer
import os
import re
from nltk import pos_tag
from definitions import *

lemma = WordNetLemmatizer()
stopWords = stopwords.words('english')
stopWords.extend(('go','wo','get','nthe','also','re','ni','ve','ni','\r\n\r\n'))

# Function displays LDA topic output
def display_topics(model, feature_names, no_top_words):
    array = []
    for topic_idx, topic in enumerate(model.components_):
        array.append(" ".join([feature_names[i]
                               for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return array

def dfprocess_single(dataframe, no_topics, no_features, no_top_words):
    tokens_df = pd.DataFrame()
    tokens_array = []
    topic_array = []
    for index, line in dataframe.iterrows():
        text = str(line['text'])
        tokens1 = word_tokenize(str(text).lower())
        filtered1 = pos_tag(tokens1)
        filtered1 = [word for word in filtered1 if word[1] in ['NN', 'nns']]
        filtered1 = [lemma.lemmatize(word[0]) for word in filtered1]
        filtered1 = [word for word in filtered1 if word not in stopWords]
        filtered1 = [word for word in filtered1 if word not in string.punctuation]
        if len(filtered1) < 2:
            filtered1 = 'NaN'
        tokens_array.append(filtered1)
        print(len(filtered1))
    tokens_df['Tokens'] = tokens_array
    docdict = tokens_df['Tokens']
    print('Word cleaning complete')
    for key, value in docdict.items():
        if value == 'NaN':
            topic_array.append('NaN')
        else:
            tf_vectorizer = CountVectorizer(max_features=no_features, stop_words=None)
            tf = tf_vectorizer.fit_transform(value)
            tf_feature_names = tf_vectorizer.get_feature_names()
            lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=10, learning_method='online',
                                            learning_offset=50., random_state=0).fit(tf)
            topic_array.append(display_topics(lda, tf_feature_names, no_top_words))
    dataframe['Review_Topics'] = topic_array

start = default_timer()
input_path = os.path.join(project_path,'datasets','cleaned_chunks','output_9.csv')
output_path = os.path.join(project_path,'datasets','review_groups','output_9_single_review_topics.csv')
review = pd.read_csv(input_path, error_bad_lines=False, encoding="ISO-8859-1").fillna(value='NA')
review = review.sample(n=100)
dfprocess_single(review,5,2,20)
review.to_csv(output_path, index=False)
duration = default_timer() - start
print("Time to run:", duration, "seconds")
