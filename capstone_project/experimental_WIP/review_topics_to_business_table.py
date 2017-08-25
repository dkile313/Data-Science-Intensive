import pandas as pd
import os
import re

businesspath = "C:/Users/David/YELP_PROJECT/datasets/yelp_academic_dataset_business_cleaned.csv"
businessdf = pd.read_csv(businesspath, error_bad_lines=False, encoding = "ISO-8859-1").fillna(value='NA')

inpath = "C:/Users/David/YELP_PROJECT/datasets/review_groups/"
#all_topics = pd.DataFrame(columns=('business_id', '1-2_Stars topic','4-5_Stars topic'))
data_list = []
for file in os.listdir(inpath):
    file_df = pd.read_csv(inpath+file, error_bad_lines=False, encoding = "ISO-8859-1").fillna(value='NA')
    file_df = file_df.drop('1-2_Stars', 1).drop('4-5_Stars', 1).drop('1-2_Stars_count', 1).drop('4-5_Stars_count', 1)
    data_list.append(file_df)
all_topics = pd.concat(data_list)
# all_topics.columns = (['1-2_star_topics','del1','4-5_star_topics','del2','business_id'])
# all_topics = all_topics.drop('del1',1).drop('del2',1)
path2 = "C:/Users/David/YELP_PROJECT/datasets/experimental_concat.csv"
path3 = "C:/Users/David/YELP_PROJECT/datasets/experimental_business.csv"
all_topics.to_csv(path2)
newdf = pd.merge(businessdf, all_topics, on='business_id')
newdf.to_csv(path3)
#print(newdf)