'''
NOTE: CODE MUST BE RUN IN PYTHON 2.7
CODE IMPORTED FROM GITHUB USER jrivero
'''

# from definitions import *
import os

input_path = os.path.join('C:/','Users','David','YELP_PROJECT','datasets','yelp_academic_dataset_review.csv')
output_path = os.path.join('C:/','Users','David','YELP_PROJECT','datasets','chunked_reviews')


def split(filehandler, delimiter=',', row_limit=50000,
          output_name_template='output_%s.csv', output_path=output_path, keep_headers=True):
    """
    Splits a CSV file into multiple pieces.

    A quick bastardization of the Python CSV library.
    Arguments:
        `row_limit`: The number of rows you want in each output file. 10,000 by default.
        `output_name_template`: A %s-style template for the numbered output files.
        `output_path`: Where to stick the output files.
        `keep_headers`: Whether or not to print the headers in each output file.
    Example usage:

        >> from toolbox import csv_splitter;
        >> csv_splitter.split(open('/home/ben/input.csv', 'r'));

    """
    import csv
    reader = csv.reader(filehandler, delimiter=delimiter)
    current_piece = 1
    current_out_path = os.path.join(
        output_path,
        output_name_template % current_piece
    )
    current_out_writer = csv.writer(open(current_out_path, 'wb'), delimiter=delimiter)
    current_limit = row_limit
    if keep_headers:
        headers = reader.next()
        current_out_writer.writerow(headers)
    for i, row in enumerate(reader):
        if i + 1 > current_limit:
            current_piece += 1
            current_limit = row_limit * current_piece
            current_out_path = os.path.join(
                output_path,
                output_name_template % current_piece
            )
            current_out_writer = csv.writer(open(current_out_path, 'wb'), delimiter=delimiter)
            if keep_headers:
                current_out_writer.writerow(headers)
        current_out_writer.writerow(row)


split(open(input_path, 'r'));
