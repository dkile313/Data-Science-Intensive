import re
from definitions import *

def business_cleaning():
    # Removes unnecessary variables and keeps only restaurants in North America
    input_path = os.path.join(project_path, 'datasets', 'yelp_academic_dataset_business.csv')
    output_path = os.path.join(project_path, 'datasets', 'yelp_academic_dataset_business_cleaned.csv')
    business = pd.read_csv(input_path, error_bad_lines=False, encoding="ISO-8859-1").fillna(value='NA')
    business = business.drop('neighborhood', 1).drop('is_open', 1)
    business = business[business['longitude'] < (-65)]
    business = business[business['categories'].str.contains("Restaurant")]
    business = business[business['review_count'] > 9]
    business.to_csv(output_path)

# Loop over files in directory by path, sorted in numeric order
def review_cleaning():
    '''Cleans the review files by removing unneeded variables and only keeping reviews for businesses in the cleaned
    business dataset'''
    business_path = os.path.join(project_path, 'datasets', 'yelp_academic_dataset_business_cleaned.csv')
    business = pd.read_csv(business_path, error_bad_lines=False, encoding="ISO-8859-1").fillna(value='NA')
    input_path = os.path.join(project_path, 'datasets', 'chunked_reviews/')
    output_path = os.path.join(project_path, 'datasets', 'cleaned_chunks/')
    cols = ('user_id', 'review_id', 'text', 'business_id', 'stars', 'date', 'type')
    # Init lastbusiness variables
    last_business1 = pd.DataFrame(columns=cols)
    last_business2 = pd.DataFrame(columns=cols)
    for file in sorted(os.listdir(input_path), key=lambda x: (int(re.sub('\D', '', x)), x)):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            review_1 = pd.read_csv(input_path + filename, error_bad_lines=False, encoding="ISO-8859-1").fillna(value='NA')
            # drop unneeded columns
            review_1 = review_1.drop('funny', 1).drop('cool', 1).drop('useful', 1)
            # only keep rows which match business_id from the business dataframe
            review_1 = (review_1[review_1['business_id'].isin(business['business_id'])])
            # code to remove the last business from this csv and add the last business from the previous csv to this one
            # this prevents businesses from being split between two csvs
            last_line = review_1.tail(1)
            last_business2 = review_1[(review_1['business_id'].isin(last_line['business_id']))]
            # concat the previous csv's lastbusiness to the current csv, then update lastbusiness2 for use in the next loop
            pd.concat([review_1, last_business1])
            last_business1 = last_business2
            review_1.to_csv(output_path + filename)

business_cleaning()
print("Business dataset cleaned")
review_cleaning()
print("Review datasets cleaned")