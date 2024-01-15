
import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import ast 

#will take 10
url_base = 'https://zenodo.org/record/7133351#.Y2PJA4LMJJU'
url = url_base
response = requests.get(url)
content = response.content
          #restructure url data for readability
soup = BeautifulSoup(content, "html.parser")

v = soup.find_all('link', href=True)
for a in v:
  if "https://zenodo.org/record/7133351/files/covid19_2" in a['href']:
    os.system("wget " + a['href'] + " -P ../data/wget/")

files = os.listdir("../data/wget/")
unique_countries = []
#Trim files to remove files with a Null location and convert to csv
for file in files:
    #if file + '.csv' in files:
    #    continue
    if '.json.csv' in file and '.swp' not in file:
        df = pd.read_csv(file)
        ds = df['location'].tolist()
        new_list = []
        for item in ds:
            new_list.append(ast.literal_eval(item))
        df = pd.DataFrame(new_list)
        countr = df['country'].tolist()

        dic = Counter(countr)
        unique_countries.append([dict(dic), file])
        try:
            if '.csv' in file:
                print(file)
            else:
                df = pd.read_json(file, lines=True)
                df = df[df['location'].notna()]
                df.to_csv(file+'.csv')
            ds = df['location'].tolist()
            print(len(ds))
            df = pd.DataFrame(ds)
            countr = df['country'].tolist()
            dic = Counter(countr)
        except:
            continue
location1 = pd.DataFrame()
location1['unique_countries'] = unique_countries
location1.to_csv('../data/unique_countries1.csv')



#Generate file concationation
list_dict = []
date1 = []
week = []
year = []
date2 = []
for item in location1['unique_countries']:
  item = ast.literal_eval(item)
  list_dict.append(item[0])
  date_item = datetime.strptime(item[1], 'covid19_%Y_%m_%d.json.csv')
  date1.append(date_item)
  week.append(date_item.strftime('%W'))
  year.append(date_item.strftime('%Y'))
  date2.append(item[1])

df2 = pd.DataFrame(list_dict)
df2['date'] = date1
df2['week'] = week
df2['year'] = year
df2['file'] = date2

#view country count item by week
df2.to_csv('week_count.csv')
grouped = df2.groupby(['week', 'year'])

#generate cat commands
for name, group in grouped:
  ini_list = group['file'].tolist()
  value_next = ("{0}".format(' '.join(map(str, ini_list))))
  os.system('cat', value_next, '>', '../data/cat/'+''.join(name)+'.csv')


files = os.listdir("../data/cat")
unique_countries = []

un_list =pd.read_csv('../data/WPP2022_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT_REV1.csv')
un_l = un_list['Country'].tolist()

id_dict = {}
for country in un_l:
    id_dict[country] = []
    id_dict[(country+'_dt')] = []
id_list = []
country_l = []
date_l = []
#loop through csv week files for sampling
for file in files:
    if '.csv' in file and '.swp' not in file and 'pull' not in file:
        print(file)
        df = pd.read_csv(file)
        for item in un_l:
            country_l.append(item)
            df1 = df[df['location'].str.contains(item +"',")]
            if len(df1) > 5:
                id_list.extend(df1.sample(n=5)['tweet_id'].tolist())
                date_l.extend(df1.sample(n=5)['date'].tolist())
            elif len(df1) > 1:
                id_list.extend(df1.sample(n=2)['tweet_id'].tolist())
                date_l.extend(df1.sample(n=2)['date'].tolist())
            elif len(df1) > 0:
                id_list.extend(df1.sample(n=1)['tweet_id'].tolist())
                date_l.extend(df1.sample(n=1)['date'].tolist())

final_df = pd.DataFrame()
try:
    new_list = [str(i) for i in id_list]
    final_df['tweet_id'] = new_list
    final_df['country'] = country_l
    final_df.to_csv('../data/hydrator_input.csv')
except:
    print(id_list)
    new_list = [str(i) for i in id_list]
    final_df['tweet_id'] = new_list
    final_df.to_csv('../data/hydrator_input.csv')