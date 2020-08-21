# http://docs.tweepy.org/en/latest/
import tweepy
import Credentials

auth = tweepy.OAuthHandler(Credentials.CONSUMER_TOKEN, CREDENTIALS.CONSUMER_SECRET)
api = tweepy.API(auth)

haikus = []
for tweet in tweepy.Cursor(api.search, q='#twaiku', rpp=100).items():
    print(tweet)
    haikus.append(tweet)
with open("data/twitter_haikus.txt", "w") as infile:
	infile.writelines(haikus)