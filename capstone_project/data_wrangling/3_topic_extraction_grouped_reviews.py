'''
Functions and loop for topic extraction of review files using LDA
'''

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from timeit import default_timer
from nltk import pos_tag
from definitions import *

pd.options.display.max_colwidth = 50
lemma = WordNetLemmatizer()
stopWords = stopwords.words('english')
stopWords.extend(('go','wo','get','nthe','also','re','ni','ve','ni','\r\n\r\n'))


# Function concatenates all 1-2 star and 3-4 star reviews for each business
def reviewgroups(input_df, output_df):
    business_array = []
    low_score_dict = {}
    high_score_dict = {}
    for index, line in input_df.iterrows():
        if line['business_id'] not in business_array:
            business_array.append(line['business_id'])
            low_score_dict[line['business_id']] = []
            high_score_dict[line['business_id']] = []
        if line['stars'] in [1,2]:
            low_score_dict[line['business_id']].append(line['text'])
        if line['stars'] in [4,5]:
            high_score_dict[line['business_id']].append(line['text'])
    output_df['business_id'] = business_array
    for index, line in output_df.iterrows():
        line['1-2_Stars'] = low_score_dict[line['business_id']]
        line['4-5_Stars'] = high_score_dict[line['business_id']]
        line['1-2_Stars_count'] = len(low_score_dict[line['business_id']])
        line['4-5_Stars_count'] = len(high_score_dict[line['business_id']])
    col_array = ['1-2_Stars', '4-5_Stars']
    for column in col_array:
        for index, line in output_df.iterrows():
            if len(line[column]) == 0:
                line[column] = 'NaN'


# Function displays LDA topic output
def display_topics(model, feature_names, no_top_words):
    array = []
    for topic_idx, topic in enumerate(model.components_):
        array.append(" ".join([feature_names[i]
                               for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return array

# Function filters text and performs LDA on the filtered output
def dfprocess(dataframe, no_topics, no_features, no_top_words):
    tokens_df = pd.DataFrame()
    col_array = ['1-2_Stars', '4-5_Stars']
    for column in col_array:
        tokens_array = []
        topic_array = []
        for index, line in dataframe.iterrows():
            text = str(line[column])
            if text == 'NaN':
                tokens_array.append('NaN')
            elif line[column+'_count'] < 3:
                tokens_array.append('NaN')
            else:
                tokens1 = word_tokenize(str(text).lower())
                filtered1 = pos_tag(tokens1)
                filtered1 = [word for word in filtered1 if word[1] in ['NN', 'nns']]
                filtered1 = [lemma.lemmatize(word[0]) for word in filtered1]
                filtered1 = [word for word in filtered1 if word not in stopWords]
                filtered1 = [word for word in filtered1 if word not in set(string.punctuation)]
                tokens_array.append(filtered1)
        tokens_df[column] = tokens_array
        docdict = tokens_df[column]
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
        dataframe[column + ' topic'] = topic_array

# Code for running LDA on one file.  This was used to generate the sample for ontology creation.
input_path = os.path.join(project_path,'datasets','cleaned_chunks','output_9.csv')
output_path = os.path.join(project_path,'datasets','review_groups','output_9_topics_2.csv')
review = pd.read_csv(input_path, error_bad_lines=False, encoding="ISO-8859-1").fillna(value='NA')
review_grouped = pd.DataFrame(columns=('business_id', '1-2_Stars', '1-2_Stars_count','4-5_Stars', '4-5_Stars_count'))
reviewgroups(review, review_grouped)
dfprocess(reviewgroups, 3, 100, 10)
reviewgroups.to_csv(output_path, index=False)


# Below code is for potential looping over all files if needed.  Current ontology is based on a single file.
'''
input_path = "C:/Users/David/YELP_PROJECT/datasets/cleaned_chunks/" # input path for cleaned reviews directory
output_path = "C:/Users/David/YELP_PROJECT/datasets/review_groups/" # output path for topic-extracted reviews directory
counter = 1
for file in sorted(os.listdir(input_path), key=lambda x: (int(re.sub('\D', '', x)), x)):
    start = default_timer()
    review = pd.read_csv(input_path + file, error_bad_lines=False, encoding="ISO-8859-1").fillna(value='NA')
    review_grouped = pd.DataFrame(columns=('business_id', '1-2_Stars', '1-2_Stars_count','4-5_Stars', '4-5_Stars_count'))
    reviewgroups(review, review_grouped)
    # review_grouped = review_grouped.sample(n=50)
    dfprocess(review_grouped, 3, 100, 10)
    review_grouped.to_csv(output_path + 'Topic_' + file, index=False)
    duration = default_timer() - start
    print("Loop",counter,"complete. Time to run:", duration, "seconds")
    # print(memorydf)
    counter+=1
    time.sleep(3)
'''