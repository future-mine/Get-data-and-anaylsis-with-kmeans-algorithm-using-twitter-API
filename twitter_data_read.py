import re
import json
import xml
import numpy as np
from collections import Counter
from TwitterAPI import TwitterAPI # in case you need to install this package, see practical 6
from sklearn.cluster import KMeans

import requests

# disabling urllib3 warnings
requests.packages.urllib3.disable_warnings()

import matplotlib.pyplot as plt 
import numpy

# %matplotlib inline

#If you need add any additional packages, then add them below


keywords = ["pizza", "bear", "wine"]

# Twitter API credentials 
CONSUMER_KEY = 'KhIdbBLLdkEdhNJyxkEk5XUIQ'
CONSUMER_SECRET =  'yjZgdGDOnbdNrq4pfUwcrDpo4A5KWMRSyMRGWzLjrZkvw4XAEH'
OAUTH_TOKEN =  '40451334-aumqCFTmJOr9B6FBEyWwvixYqwrVhzRLJdxgGCToW'
OAUTH_TOKEN_SECRET =  'QzpgKj5HILsxMPlreDUDtJgjp7cBgsX4EYJQcpYm5awSC'

# Authonticating with your application credentials
api = TwitterAPI(CONSUMER_KEY,
                 CONSUMER_SECRET,
                 OAUTH_TOKEN,
                 OAUTH_TOKEN_SECRET) #INSERT YOUR CODE HERE

# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
# geo coordinations of the desired place
PLACE_LAT = 43.000000
PLACE_LON = -75.000000
PLACE_RAD = 100




def retrieve_tweets(api, keyword, batch_count, total_count, latitude=[], longitude=[], radius=0):
    """
    collects tweets using the Twitter search API
    
    api:         Twitter API instance
    keyword:     search keyword
    batch_count: maximum number of tweets to collect per each request
    total_count: maximum number of tweets in total
    """
    
    # the collection of tweets to be returned
    tweets_unfiltered = []
    tweets = []
    
    # the number of tweets within a single query
    batch_count = str(batch_count)
    
    '''
    You are required to insert your own code where instructed to perform the first query to Twitter API.
    Hint: revise the practical session on Twitter API on how to perform query to Twitter API.
    '''
    # per the first query, to obtain max_id_str which will be used later to query sub
    resp = api.request('search/tweets', {
                                         'q': keyword,#INSERT YOUR CODE
                                         'count':'US', #INSERT YOUR CODE
                                         'lang':'en',
                                         'result_type':'recent',
                                         'geocode':'{},{},{}mi'.format(latitude, longitude, radius)
                                         })
    
    # store the tweets in a list

    # check first if there was an error
    if ('errors' in resp.json()):
        errors = resp.json()['errors']
        if (errors[0]['code'] == 88):
            print('Too many attempts to load tweets.')
            print('You need to wait for a few minutes before accessing Twitter API again.')
    
    if ('statuses' in resp.json()):
        tweets_unfiltered += resp.json()['statuses']
        print(tweets_unfiltered)
        tweets = [tweet for tweet in tweets_unfiltered if ((tweet['retweeted'] != True) and ('RT @' not in tweet['text']))]
    
        # find the max_id_str for the next batch
        ids = [tweet['id'] for tweet in tweets_unfiltered]
        print(ids)
        max_id_str = str(min(ids))

        # loop until as many tweets as total_count is collected
        number_of_tweets = len(tweets)
        while number_of_tweets < total_count:

            resp = api.request('search/tweets', {
                                             'q':keyword, #INSERT YOUR CODE                                             'count':'US', #INSERT YOUR CODE
                                             'lang':'en',
                                             'result_type':'recent',  #INSERT YOUR CODE
                                             'max_id': max_id_str,
                                             'geocode': '{},{},{}mi'.format(latitude, longitude, radius)#INSERT YOUR CODE
            })

            if ('statuses' in resp.json()):
                tweets_unfiltered += resp.json()['statuses']
                tweets = [tweet for tweet in tweets_unfiltered if ((tweet['retweeted'] != True) and ('RT @' not in tweet['text']))]
 
                ids = [tweet['id'] for tweet in tweets_unfiltered]
                max_id_str = str(min(ids))
            
                number_of_tweets = len(tweets)
        
            print("{} tweets are collected for keyword {}. Last tweet created at {}".format(number_of_tweets, 
                                                                                    keyword, 
                                                                                    tweets[number_of_tweets-1]['created_at']))
    return tweets





# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
'''
Your task is to write the code to perform three function calls, each corresponds to one keyword. 
And, you are required to collect at least 200 tweets for each key word.
'''

# Collecting the tweets for three assigned keywords, 
# Your function call should look like this:  retrieve_tweets(api,'keyword',single_count,total_count)

k1_tweets = retrieve_tweets(api,keywords[0],0,50,PLACE_LAT, PLACE_LON, PLACE_RAD)#INSERT YOUR CODE HERE
k2_tweets = retrieve_tweets(api,keywords[1],0,50,PLACE_LAT, PLACE_LON, PLACE_RAD)#INSERT YOUR CODE HERE
k3_tweets = retrieve_tweets(api,keywords[2],0,50,PLACE_LAT, PLACE_LON, PLACE_RAD)#INSERT YOUR CODE HERE

# PLEASE NOTE THAT IF YOU RUN THIS CELL, IT MIGHT TAKE A WHILE TO DOWNLOAD ALL THE TWEETS REQUIRED.
# MAKE SURE THAT YOU WAIT UNTILL THE CELL FINISHES RUNNING.




# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
'''
Write your code to print the number of tweets have been collected for each keyword
'''
# INSERT YOUR CODE HERE
print('pizza: ', len(k1_tweets))
print('bear: ', len(k2_tweets))
print('beaf: ', len(k1_tweets))





# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
'''
Write your code to print out all fields of the first tweet
Hint: You might want to use method keys() of the dictionary
'''
# INSERT YOUR CODE HERE

firsttweet1 = k1_tweets[0]
firsttweet2 = k2_tweets[0]
firsttweet3 = k3_tweets[0]
print(firsttweet1.keys())
print(firsttweet2.keys())
print(firsttweet3.keys())

'''
Write your code to print out the text of the first  tweet collected for each keyword.
'''

print("\nThe text of the first tweet for \"{}\":\n".format(keywords[0]))
# INSERT YOUR CODE HERE
print(firsttweet1['text'])


print("\nThe text of the first tweet for \"{}\":\n".format(keywords[1]))
# INSERT YOUR CODE HERE
print(firsttweet2['text'])


print('\nThe text of the first tweet for \"{}\":\n'.format(keywords[2]))
# INSERT YOUR CODE HERE
print(firsttweet3['text'])





def save_to_json(obj, filename):
    """
    saves a list of dictionaries into a json file
    
    obj: list of dictionaries
    filename: filename
    """
    with open(filename, 'w') as fp:
        json.dump(obj, fp, indent=4, sort_keys=True)   




# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
'''
Use the save_to_json() function defined above, for the collection of tweets 
you have crawled for each keyword, save them into a file named w.json where w is the keyword, taken from keywords list.
'''
# saving the tweets in three json files, one for each keyword
#INSERT YOUR CODE HERE
save_to_json(k1_tweets, keywords[0]+'.json')
save_to_json(k2_tweets, keywords[1]+'.json')
save_to_json(k3_tweets, keywords[2]+'.json')









