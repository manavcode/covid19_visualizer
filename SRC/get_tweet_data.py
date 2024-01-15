import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
import numpy as np
from collections import Counter

files = os.listdir("../data/wget/")
unique_countries = []
pull_df = pd.read_csv('../data/batch.csv')

#filter for only tweets in english
pull_df = pull_df[pull_df['lang']=='en']
pull_df['id_2'] = pull_df['tweet_url'].str[-19:]

#acquire id through hydrated id column
id_list = pull_df['id'].to_list()
#acquire id through tweet link
id2_list = pull_df['id_2'].to_list()

#generate id and country list
ids_ = []
country_ = []
for file in files:
    if '.csv' in file and '.swp' not in file and 'pull' not in file and 'batch' not in file:
        print(file)
        df = pd.read_csv(file)
        new_df = df.loc[((df['tweet_id'].isin(id_list)) | (df['tweet_id'].isin(id2_list)))]
        if len(new_df) > 0:
            ids_.extend(new_df['tweet_id'].to_list())
            country_.extend(new_df['location'].to_list())

#create country id dictionary
country_dict = {}
for i in range(len(ids_)):
    country_dict[str(ids_[i])] = country_[i]

#return country and id csv only           
final_df = pd.DataFrame()
final_df['tweet_id'] = ids_
final_df['country'] = country_
final_df.to_csv('second_batch_country.csv')

#use dictionary to pull correct order of countries per id
fail = []
final_order = []
for item in id_list:
    if str(item) in country_dict.keys():
        final_order.append(country_dict[str(item)])
    else:
        final_order.append("not found")
        fail.append(item)

print(fail)
pull_df['COUNTRY'] = final_order

#add the country column to original data set
pull_df.to_csv('../data/pull_final.csv')