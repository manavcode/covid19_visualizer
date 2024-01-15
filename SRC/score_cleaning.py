#pip install pycountry
import pandas as pd
import datetime
import pandas as pd
import numpy as np
import pycountry
from datetime import datetime,timedelta


# Reading sample data using pandas DataFrame
csvfile = '../data/full_processed.csv'
df = pd.read_csv(csvfile)


df['week'] = pd.to_datetime(df['full_date']).dt.isocalendar().week
df['year'] = pd.to_datetime(df['full_date'])
df['year'] = pd.DatetimeIndex(df['year']).year
df['month'] = pd.DatetimeIndex(df['year']).month

df['week'] = df['week'].replace([53], 1)


def change_week(week, month, year):
    if week == 52 and month == 1 and year == 2021 or year == 2022:
        week = 1
    if week == 53 and month == 1 and year == 2021 or year == 2022:
        week = 1
    return week


df['true_week'] = df.apply(lambda x: change_week(week=x['week'], month=x['month'], year=x['year']), axis=1)

df.loc[df.year == 2021, 'true_week'] = df['true_week'] + 52
df.loc[df.year == 2022, 'true_week'] = df['true_week'] + 104

data = df

def true_week_time(week):
    date_and_time = datetime(2020,1 , 1)
    time_change = timedelta(weeks=week)
    return date_and_time + time_change

data["country_cor"] = data["COUNTRY.1"].str.strip()

country_list = [x.name for x in list(pycountry.countries)]
unique_weeks = data["true_week"].unique()
unique_country = data["country_cor"].unique()

country_dict_new = {}

for week in unique_weeks:
    country_dict_new[week] = {}
    for country in country_list:
        country_dict_new[week][country] = {"vader_Sentiment_Score":0, "weighted": 0, "fake": 0, "confidence": 0}

country_dict = {}

for country in country_list:
    df = data[data["country_cor"]== country]
    
    if not (df.empty):
        country_dict[country] = {}
        weeks = df["true_week"].values
        for week in weeks:
            country_dict[country][week] = {"vader_Sentiment_Score":0, "weighted": 0, "fake": 0, "confidence": 0}

gb_df = data.groupby(["country_cor","actual_week"]).agg({'vader_Sentiment_Score': 'mean', 'weighted': 'mean',
                                                      'fake': 'sum', 'confidence': 'mean',})
        
for country in country_dict:
    for week in country_dict[country]:
        country_dict[country][week]["vader_Sentiment_Score"] = (gb_df.loc[country , week]["vader_Sentiment_Score"])
        country_dict[country][week]["weighted"] = (gb_df.loc[country , week]["weighted"])
        country_dict[country][week]["fake"] = (gb_df.loc[country , week]["fake"])
        country_dict[country][week]["confidence"] = (gb_df.loc[country , week]["confidence"])  

for country in country_dict:
    for week in country_dict[country]:
        country_dict_new[week][country]["vader_Sentiment_Score"] = (gb_df.loc[country , week]["vader_Sentiment_Score"])
        country_dict_new[week][country]["weighted"] = (gb_df.loc[country , week]["weighted"])
        country_dict_new[week][country]["fake"] = (gb_df.loc[country , week]["fake"])
        country_dict_new[week][country]["confidence"] = (gb_df.loc[country , week]["confidence"])

output_dict = {"week":[],"code":[], "country": [], 'vader_Sentiment_Score':  [], "weighted": [], 'fake':  [], "confidence": []}
for week in country_dict_new:
    for country in country_dict_new[week]:
        output_dict["week"].append(week)
        output_dict["code"].append(pycountry.countries.get(name=country).alpha_3)
        output_dict["country"].append(country)
        output_dict["vader_Sentiment_Score"].append(country_dict_new[week][country]["vader_Sentiment_Score"])
        output_dict["weighted"].append(country_dict_new[week][country]["weighted"])
        output_dict["fake"].append(country_dict_new[week][country]["fake"])
        output_dict["confidence"].append(country_dict_new[week][country]["confidence"])

output_df = pd.DataFrame(output_dict)
output_df = output_df[output_df['week'].notna()]

output_df["true_week_time"] = output_df.apply(lambda x: true_week_time( week = x['week']), axis = 1)

output_df.rename({'vader_Sentiment_Score': 'Sentiment Score', 'weighted': 'Weighted Sentiment Score by Retweets', 'Fake':'Fake Tweets Count', 'confidence':'Illegitimacy Confidence'}, axis=1) 

output_df.to_csv('../data/cleaned_agg_scores.csv')