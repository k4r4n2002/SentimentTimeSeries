import snscrape.modules.twitter as sntwitter
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

query="(#recession) min_retweets:10 lang:en until:2022-06-30 since:2010-01-01 -filter:links -filter:replies"
tweets=[]
limit=50000


for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets)==limit:
        break
    else:
        tweets.append([tweet.date,tweet.user.username,tweet.content])

df=pd.DataFrame(tweets,columns=['Date','User','Tweet'])
print(df)
df.to_csv('Recession3.csv')