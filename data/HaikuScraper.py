# http://docs.tweepy.org/en/latest/
import tweepy
import Credentials

auth = tweepy.OAuthHandler(Credentials.CONSUMER_TOKEN, Credentials.CONSUMER_SECRET)
api = tweepy.API(auth)

haikus = []
for tweet in tweepy.Cursor(api.search, q='#twaiku').items():
	haikus.append(tweet.text)

with open("twitter_haikus.txt", "w", errors="ignore") as infile:
	infile.writelines(haikus)