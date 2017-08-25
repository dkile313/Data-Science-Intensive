import os
import pandas as pd
import re



business_path = "C:/Users/David/YELP_PROJECT/datasets/yelp_academic_dataset_business_cleaned.csv"  # Clean business path
path = "C:/Users/David/YELP_PROJECT/datasets/yelp_academic_dataset_review.csv"  # input path for chunked reviews
path2 = "C:/Users/David/YELP_PROJECT/datasets/review_cleaned.csv"  # output path for chunked reviews
business = pd.read_csv(business_path, error_bad_lines=False, encoding="ISO-8859-1").fillna(value='NA')
reviews_len = 0 # review length counter

# Init lastbusiness variables
lastbusiness1 = pd.DataFrame(columns=('user_id','review_id','text','business_id','stars','date','type'))
lastbusiness2 = pd.DataFrame(columns=('user_id','review_id','text','business_id','stars','date','type'))


review_1 = pd.read_csv(path, error_bad_lines=False, encoding="ISO-8859-1").fillna(value='NA')
# drop unneeded columns
review_1 = review_1.drop('funny', 1).drop('cool', 1).drop('useful', 1)
# only keep rows which match business_id from the business dataframe
review_1 = (review_1[review_1['business_id'].isin(business['business_id'])])
print(path, 'new length =', len(review_1))
review_1.to_csv(path2)
