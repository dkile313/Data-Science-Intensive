###NOTE: RUN IN PYTHON 2.7###

import csv
import glob
import os

input_path = "C:/Users/David/YELP_PROJECT/datasets/cleaned_chunks/"
output_file = 'C:/Users/David/YELP_PROJECT/datasets/reviews_cleaned.csv'

filewriter = csv.writer(open(output_file,'wb'))
file_counter = 0
for input_file in glob.glob(os.path.join(input_path,'*.csv')):
        with open(input_file,'rU') as csv_file:
                filereader = csv.reader(csv_file)
                if file_counter < 1:
                        for row in filereader:
                                filewriter.writerow(row)
                else:
                        header = next(filereader,None)
                        for row in filereader:
                                filewriter.writerow(row)
        file_counter += 1