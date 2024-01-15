import pandas as pd
import numpy as np
import io
import re
# !pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


name = "../data/final_pull.csv"


df = pd.read_csv(name)
# df = df.drop(["Unnamed: 0", "Unnamed: 0.1", "coordinates", "place", "user_name", "media", "urls", "in_reply_to_screen_name", "in_reply_to_status_id", "in_reply_to_user_id", "possibly_sensitive", "quote_id", "retweet_id", "retweet_screen_name", "source", "tweet_url", "user_created_at", "user_default_profile_image", "user_description", "user_favourites_count", "user_followers_count", "user_friends_count", "user_listed_count", "user_statuses_count", "user_time_zone", "user_urls", "label"], axis = 1)
df = df.drop(["Unnamed: 0", "Unnamed: 0.1", "label"], axis=1)
df = df.loc[df["lang"] == "en"]
df = df.rename(columns={"tweet": "text"})
df = df.dropna(subset=["text", "user_location", "created_at"]) 


df["text"] = df["text"].replace(r'https://\S+', '', regex=True)

# print(str(df["text"].head(100)))
#Removing speacial char
df["text"] = df["text"].replace('[*)@#%(&$_;:|\-^]', '', regex=True)
df["text"] = df["text"].replace('[\n]', '.', regex=True)
df["text"] = df["text"].apply(lambda x: str.lstrip(x))
df["text"] = df["text"].apply(lambda x: str.rstrip(x))
df_vader = df.copy()

tweets_to_analyze = df["text"].values.tolist()
sentiment_analyzer = SentimentIntensityAnalyzer()
sentiments = []
score = []
for tweet in tweets_to_analyze: 
  dict1 = sentiment_analyzer.polarity_scores(tweet)
  if dict1["compound"] > 0.05: 
    sentiments.append("pos")
  elif dict1["compound"] < -0.05: 
    sentiments.append("neg")
  else: 
    sentiments.append("neu")
  score.append(dict1["compound"])
df["vader_Classified_Sentiment"] = sentiments 
df["vader_Sentiment_Score"] = score
temp = pd.to_datetime(df["created_at"])

df["weighted"] = df["vader_Sentiment_Score"]*df["retweet_count"]
df["week"] = temp.dt.isocalendar().week
df["year"] = temp.dt.year
df["full_date"] = temp.dt.date
df["full_time"] = temp.dt.time

# print(len(df))
for index, row in df.iterrows():
  if row["week"] == 0:
    df["year"][index] -= 1
    df["week"][index] = 1
  if row["week"] == 53:
    df["year"][index] += 1
    df["week"][index] = 52


df.to_csv("../data/test_full_processed.csv")
