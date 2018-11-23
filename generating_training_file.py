#import pandas as pd

#data_file = open("yelp_academic_dataset_review.json", "r")
#values = json.load(data_file)
#data_file.close()

#with open("training_data.csv", "wb") as f:
#    wr = csv.writer(f)
#    keys = ["user_id","business_id","stars"]
#    for data in values:
#         for key in keys:
#               wr.writerow([data[keys[0]], data[keys[1]], data[keys[2]]])

#df = pd.read_json("./yelp_academic_dataset_review.json")
#df.head(1)

import json
import csv

keys = ["user_id","business_id","stars"]
with open("training_data.csv", "wb") as fcsv:
    wr = csv.writer(fcsv)
    wr.writerow(['user_id','business_id','stars'])
    with open('yelp_academic_dataset_review.json') as f:
        for line in f:
            data = json.loads(line)
            wr.writerow([data[keys[0]], data[keys[1]], data[keys[2]]])