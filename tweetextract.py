__author__ = 'Soumik'
import tweepy
import sys
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import pandas as pd
import matplotlib.pyplot as plt
import codecs
atoken = "1220909827-X6n5MabisHf7a1hktJ93JPLOwkenN2TU5Nisu8G"
asecret = "F0fieS5oixKZ3W97FDhV38RhR9PYgoXFpr5yGH4mSa0ku"
ckey = "mSi6hbX1lRZiWkN3cGUjaGtOR"
csecret = "fXZHAzs9j1QzoLuOFSnE7mCdAIzifBgCNUmGOUfzt6SWnAiNe9"
auth = tweepy.OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
def uprint(*objects, sep=' ', end='\n', file=sys.stdout):
    enc = file.encoding
    if enc == 'UTF-8':
        print(*objects, sep=sep, end=end, file=file)
    else:
        f = lambda obj: str(obj).encode(enc, errors='backslashreplace').decode(enc)
        print(*map(f, objects), sep=sep, end=end, file=file)
twitterApi = tweepy.API(auth)
f=open("new4.txt",'w')
def iterjsoon(f):
    for key in f.keys():
        if(isinstance(f[key],dict)):
            uprint(key,":")
            print("{")
            iterjsoon(f[key])
            print("}")
        else:
            uprint(key,":",f[key])
outfile=open('new3.txt', 'w')
class ReplyToTweet(StreamListener):
    def on_data(self, data):
        s=json.loads(data)

        print("hello")
        json.dump(data, outfile)


    def on_error(self, status):
        print (status)

if __name__ == '__main__':
    l = ReplyToTweet()
    twitterStream = Stream(auth, l)
    twitterStream.filter(track=['#sarcasm','#sarcastic'])