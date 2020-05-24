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
from scipy import spatial
# %matplotlib inline

#If you need add any additional packages, then add them below
from scipy import spatial
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures

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
PLACE_LAT = [40.0160921, -105.2812196]
PLACE_LON = [-105.2812196, 40.0160921]
PLACE_RAD = 100








# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
'''
Insert your own code where instructed to complete this function
'''
def read_json_file(filename):
    """
    reads from a json file and saves the result in a list named data
    """
    with open(filename, 'r') as fp:
    # INSERT THE MISSING PIECE OF CODE HERE
        content = fp.read()
    
    data = json.loads(content)
    return data     




# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
'''
Write three function calls to load data from three json files you have saved from Part 1.
'''

k1_tweets = read_json_file(keywords[0]+'.json')
k2_tweets = read_json_file(keywords[1]+'.json')
k3_tweets = read_json_file(keywords[2]+'.json')
print(len(k1_tweets))
print(len(k2_tweets))
print(len(k3_tweets))




# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
def is_short_tweet(tweet):
    '''
    Check if the text of "tweet" has less than 50 characters
    '''
    # INSERT YOUR CODE HERE
    if(len(tweet['text']) < 50):
        return True

    return False




# YOU ARE REQUIRED TO INSERT YOUR CODES IN THIS CELL
'''
Write your codes to remove all tweets which have less than 50 characters in variables 
k1_tweets, k2_tweets and k3_tweets and store the results in the new variables 
k1_tweets_filtered, k2_tweets_filtered and k3_tweets_filtered respectively
'''
# INSERT YOUR CODE HERE
k1_tweets_filtered = [tweet for tweet in k1_tweets if (is_short_tweet(tweet)!=True)]
k2_tweets_filtered = [tweet for tweet in k2_tweets if (is_short_tweet(tweet)!=True)]
k3_tweets_filtered = [tweet for tweet in k3_tweets if (is_short_tweet(tweet)!=True)]

# these lines below print the number of tweets for each keyword before and after filtered.
print(len(k1_tweets), len(k1_tweets_filtered))
print(len(k2_tweets), len(k2_tweets_filtered))
print(len(k3_tweets), len(k3_tweets_filtered))




# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
'''
For each keyword, print out the number of tweets that have been removed.
'''

# INSERT YOUR CODE HERE

print(len(k1_tweets)-len(k1_tweets_filtered))
print(len(k2_tweets)-len(k2_tweets_filtered))
print(len(k3_tweets)-len(k3_tweets_filtered))





# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
'''
Write your code to print out the first 5 tweets for each keyword.
You must use the variables k1_tweets_filtered, k2_tweets_filtered and k3_tweets_filtered 
which have stored the data after the filtering process for this task.

Hint: Using tweet['text'] for tweet in k1_tweets_filtered
'''

print('The first 5 tweets for \"{}\":\n'.format(keywords[0]))
# INSERT YOUR CODE HERE
k1_first5 = [k1_tweets_filtered[id]['text'] for id in range(0,5)]
print(k1_first5)

print('\nThe first 5 tweets for \"{}\":\n'.format(keywords[1]))
# INSERT YOUR CODE HERE 
k2_first5 = [k2_tweets_filtered[id]['text'] for id in range(0,5)]
print(k2_first5)


print('\nThe first 5 tweets for \"{}\":\n'.format(keywords[2]))
# INSERT YOUR CODE HERE
k3_first5 = [k3_tweets_filtered[id]['text'] for id in range(0,5)]
print(k3_first5)





def remove_non_ascii(s): return "".join(i for i in s if ord(i)<128)
def pre_process(doc):
    """
    pre-processes a doc
      * Converts the tweet into lower case,
      * removes the URLs,
      * removes the punctuations
      * tokenizes the tweet
      * removes words less that 3 characters
    """
    
    doc = doc.lower()
    # getting rid of non ascii codes
    doc = remove_non_ascii(doc)
    
    # replacing URLs
    url_pattern = "http://[^\s]+|https://[^\s]+|www.[^\s]+|[^\s]+\.com|bit.ly/[^\s]+"
    doc = re.sub(url_pattern, 'url', doc) 

    # removing dollars and usernames and other unnecessary stuff
    userdoll_pattern = "\$[^\s]+|\@[^\s]+|\&[^\s]+|\*[^\s]+|[0-9][^\s]+|\~[^\s]+"
    doc = re.sub(userdoll_pattern, '', doc)
    
    
    # removing punctuation
    punctuation = r"\(|\)|#|\'|\"|-|:|\\|\/|!|\?|_|,|=|;|>|<|\.|\@"
    doc = re.sub(punctuation, ' ', doc)
    
    part_arr = [w for w in doc.split() if len(w) > 2]
    return part_arr



tweet_k1 = k1_tweets_filtered[0]['text']
tweet_k1_processed = pre_process(tweet_k1)

print(tweet_k1)
# tweet_k1_processed is now a list of words. 
# We use ' '.join() method to join the list to a string.
print(' '.join(tweet_k1_processed))




# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
'''
Use the example above, write your code to display the first tweets stored in 
the variables k2_tweets_filtered and k3_tweets_filtered before and after they 
have been pre-processed using the function pre_process() supplied earlier.
'''

# INSERT YOUR CODE HERE
tweet_k2 = k2_tweets_filtered[0]['text']
tweet_k2_processed = pre_process(tweet_k2)

print(tweet_k2)
# tweet_k1_processed is now a list of words. 
# We use ' '.join() method to join the list to a string.
print(' '.join(tweet_k2_processed))

tweet_k3 = k3_tweets_filtered[0]['text']
tweet_k3_processed = pre_process(tweet_k3)

print(tweet_k3)
# tweet_k1_processed is now a list of words. 
# We use ' '.join() method to join the list to a string.
print(' '.join(tweet_k3_processed))





# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
'''
Write your code to pre-process and clean up all tweets 
stored in the variable k1_tweets_filtered, k2_tweets_filtered and k3_tweets_filtered using the 
function pre_process() to result in new variables k1_tweets_processed, k2_tweets_processed 
and k3_tweets_processed.
'''
# INSERT YOUR CODE HERE
k1_tweets_processed = []
for tweet in k1_tweets_filtered:
    tweet1 = pre_process(tweet['text'])
    k1_tweets_processed.append(tweet1)

k2_tweets_processed = []
for tweet in k2_tweets_filtered:
    tweet2 = pre_process(tweet['text'])
    k2_tweets_processed.append(tweet2)

k3_tweets_processed = []
for tweet in k3_tweets_filtered:
    tweet3 = pre_process(tweet['text'])
    k3_tweets_processed.append(tweet3)
# k1_tweets_processed = [pre_process(tweet) for tweet in k1_tweets_filtered]
# k2_tweets_processed = [pre_process(tweet) for tweet in k2_tweets_filtered]
# k3_tweets_processed = [pre_process(tweet) for tweet in k3_tweets_filtered]



# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
'''
Now write your code to print out the first 5 processed tweets for each keyword.
Hint: Each tweet in tweets_processed is now a list of words, not a string. 
      To print a string, you might need to use ' '.join(tweet), 
      when tweet is a processed tweet

'''

print('The first 5 processed tweets for k1_tweets_processed:')
# INSERT YOUR CODE HERE
for id in range(0,5):
    print(' '.join(k1_tweets_processed[id]))

print('\nThe first 5 processed tweets for k2_tweets_processed:')
# INSERT YOUR CODE HERE
for id in range(0,5):
    print(' '.join(k2_tweets_processed[id]))

print('\nThe first 5 processed tweets for k3_tweets_processed:')
# INSERT YOUR CODE HERE
for id in range(0,5):
    print(' '.join(k3_tweets_processed[id]))





def construct_termdoc(docs, vocab=[]):
    """
    Construct a term-by-document-matrix
    
    docs: corpus
    vocab: pre-defined vocabulary
           if not supplied it will be automatically induced from the data
    
    returns the term-by-document matrix and the vocabulary of the passed corpus
    """
    
    # vocab is not passed
    if vocab == []:
        vocab = set()
        termdoc_sparse = []

        for doc in docs:       
            # computes the frequencies of doc
            doc_sparse = Counter(doc)    
            termdoc_sparse.append(doc_sparse)
            
            # update the vocab
            vocab.update(doc_sparse.keys())  

        vocab = list(vocab)
        vocab.sort()
    
    else:
        termdoc_sparse = []        
        for doc in docs:
            termdoc_sparse.append(Counter(doc))
            

    n_docs = len(docs)
    n_vocab = len(vocab)
    termdoc_dense = np.zeros((n_docs, n_vocab), dtype=int)

    for j, doc_sparse in enumerate(termdoc_sparse):
        for term, freq in doc_sparse.items():
            try:
                termdoc_dense[j, vocab.index(term)] = freq
            except:
                pass
            
    return termdoc_dense, vocab



# In the function construct_termdoc(), a function "set" is used. Learn what this function does 
# and explain its role in the function construct_termdoc().
# YOU ARE REQUIRED TO INSERT YOUR COMMENT IN THIS CELL
#
#




'''
compute the term-by-document matrix and the the dictionary from the collection of 
tweets collected for the first keyword
'''
k1_termdoc, k1_vocab = construct_termdoc(k1_tweets_processed)

print(k1_tweets_processed)
# print out the term-by-document matrix
print(k1_termdoc)
# print out the first 5 vocabulary entries
print(' '.join(k1_vocab[-5:]))  # print out only the first 5 vocabulary entries

# visualise the term-by-document matrix
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(k1_termdoc)
ax.set_xlabel('term (vocabulary)')
ax.set_ylabel('documents (tweets)')
ax.set_title('Term-by-Document matrix from tweets collected for keyword \"{}\"'.format(keywords[0]))

plt.show()




# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL

def Euclidean_distance(x,y):
    '''
    Compute and return the Euclidean distance between two vectors x and y
    '''
    # INSERT YOUR CODE HERE
    sub = x - y
    if sub.dot(sub) == 0:
        dist = 0
    else:
        dist = np.linalg.norm(sub)
    return dist


def cosine_distance(x,y):
    '''
    Compute and return the cosine distance between two vectors x and y
    '''
    # INSERT YOUR CODE HERE
    if x.dot(y) == 0:
        result = 0
    else:
        result = 1 - spatial.distance.cosine(x, y)
    return result





def compute_distance_matrices(termdoc):
    demens = len(termdoc)
    print(demens)
    # INSERT YOUR CODE HERE
    euclidean_distance_matrix = np.zeros([demens,demens])
    cosine_distance_matrix = np.zeros([demens, demens])
    for i in range(0, demens):
        for j in range(i, demens):
            euclidean_distance_matrix[i, j] = Euclidean_distance(termdoc[i], termdoc[j])
            cosine_distance_matrix[i, j] = cosine_distance(termdoc[i], termdoc[j])
            if(i != j):
                euclidean_distance_matrix[j, i] = Euclidean_distance(termdoc[i], termdoc[j])
                cosine_distance_matrix[j, i] = cosine_distance(termdoc[i], termdoc[j])
    return euclidean_distance_matrix, cosine_distance_matrix




# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL

# compute the distance matrices for k1_termdoc using the function "compute_distance_matrices"
# INSERT YOUR CODE HERE
k1_euclidean_distance_matrix, k1_cosine_distance_matrix = compute_distance_matrices(k1_termdoc)

print(k1_euclidean_distance_matrix)
print(k1_cosine_distance_matrix)

# Visualise the distance matrices for this keyword
# Hint: use imshow() and colorbar() functions
# INSERT YOUR CODE HERE
plt.subplot(1, 2, 1)
plt.imshow(k1_euclidean_distance_matrix, cmap='jet', aspect='auto')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(k1_cosine_distance_matrix,  cmap='jet', aspect='auto')
plt.colorbar()
plt.show()





# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
'''
Write your codes to compute the term-by-document matrix and the vocabulary for tweets stored 
in k2_tweets_processed
'''
# INSERT YOUR CODE HERE
k2_termdoc, k2_vocab = construct_termdoc(k2_tweets_processed)

'''
Write your code print out the first 5 vocabularies 
'''
# INSERT YOUR CODE HERE
print(k2_tweets_processed)
# print out the term-by-document matrix
print(k2_termdoc)
# print out the first 5 vocabulary entries
print(' '.join(k2_vocab[-5:]))  # print out only the first 5 vocabulary entries


'''
Write your code to visualise the term-by-document matrix
'''
# INSERT YOUR CODE HERE
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(k2_termdoc)
ax.set_xlabel('term (vocabulary)')
ax.set_ylabel('documents (tweets)')
ax.set_title('Term-by-Document matrix from tweets collected for keyword \"{}\"'.format(keywords[1]))

plt.show()



# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL

# compute the distance matrix for k1_termdoc using the function "compute_euclidean_distance_matrix"
# INSERT YOUR CODE HERE
k2_euclidean_distance_matrix, k2_cosine_distance_matrix = compute_distance_matrices(k2_termdoc)

# Visualise the distance matrix for this keyword
# Hint: use imshow() and colorbar() functions
# INSERT YOUR CODE HERE
plt.subplot(1, 2, 1)
plt.imshow(k2_euclidean_distance_matrix, cmap='jet', aspect='auto')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(k2_cosine_distance_matrix, cmap='jet', aspect='auto')
plt.colorbar()
plt.show()



# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
'''
Write your codes to compute the term-by-document matrix and the vocabulary for tweets stored 
in k3_tweets_processed
'''
# INSERT YOUR CODE HERE
k3_termdoc, k3_vocab = construct_termdoc(k3_tweets_processed)


'''
Write your code print out the first 5 vocabularies 
'''
# INSERT YOUR CODE HERE

print(' '.join(k3_vocab[-5:])) 


'''
Write your code to visualise the term-by-document matrix
'''
# INSERT YOUR CODE HERE
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(k3_termdoc)
ax.set_xlabel('term (vocabulary)')
ax.set_ylabel('documents (tweets)')
ax.set_title('Term-by-Document matrix from tweets collected for keyword \"{}\"'.format(keywords[2]))

plt.show()






# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL

# compute the distance matrices for k1_termdoc using the function "compute_distance_matrices"
# INSERT YOUR CODE HERE
k3_euclidean_distance_matrix, k3_cosine_distance_matrix = compute_distance_matrices(k3_termdoc)

# Visualise the distance matrix for this keyword
# Hint: use imshow() and colorbar() functions
# INSERT YOUR CODE HERE
plt.subplot(1, 2, 1)
plt.imshow(k3_euclidean_distance_matrix, cmap='jet', aspect='auto')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(k3_cosine_distance_matrix, cmap='jet', aspect='auto')
plt.colorbar()
plt.show()






# Compare the ranges of the values for cosine and Euclidean distances. 
# Explain why the ranges are different. Explain why cosine distance 
# is more convenient than Euclidean distance for text analysis.
#
# YOU ARE REQUIRED TO INSERT YOUR COMMENT IN THIS CELL
#










print('Dimension of the term-by-document matrix for keyword \"{}\":'.format(keywords[0]))
print('{} x {}\n'.format(k1_termdoc.shape[0],k1_termdoc.shape[1]))

print('Dimension of the term-by-document matrix for keyword \"{}\":'.format(keywords[1]))
print('{} x {}\n'.format(k2_termdoc.shape[0],k2_termdoc.shape[1]))

print('Dimension of the term-by-document matrix for keyword \"{}\":'.format(keywords[2]))
print('{} x {}\n'.format(k3_termdoc.shape[0],k3_termdoc.shape[1]))




all_tweets_processed = k1_tweets_processed + k2_tweets_processed + k3_tweets_processed




# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
'''
Write your codes to compute the term-by-document matrix and the vocabulary for all tweets stored 
in all_tweets_processed
'''

all_termdoc, all_vocab = construct_termdoc(all_tweets_processed)

'''
Write your code print out the first 5 vocabularies 
'''
# INSERT YOUR CODE HERE
print(' '.join(all_vocab[-5:]))

'''
Write your code to visualise the term-by-document matrix
'''
# INSERT YOUR CODE HERE
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(all_termdoc)
ax.set_xlabel('term (vocabulary)')
ax.set_ylabel('documents (tweets)')
ax.set_title('Term-by-Document matrix from tweets collected for keyword \"{}\"'.format(keywords[2]))

plt.show()




# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL

# compute the Euclidean distance matrix using compute_euclidean_distance_matrix() function

all_euclidean_distances, all_cosine_distances =  compute_distance_matrices(all_termdoc)


# Visualise the distance matrix for this keyword
# INSERT YOUR CODE HERE
plt.subplot(1, 2, 1)
plt.imshow(all_euclidean_distances, cmap='jet', aspect='auto')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(all_cosine_distances, cmap='jet', aspect='auto')
plt.colorbar()
plt.show()




# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
# 1. Your task is to produce a scatter plot of Euclidean vs cosine distance for all tweets.
# INSERT YOUR CODE HERE
print('{} x {}\n'.format(all_euclidean_distances.shape[0],all_euclidean_distances.shape[1]))
plt.scatter(all_euclidean_distances, all_cosine_distances)
plt.xlabel('Euclidean distances') 
plt.ylabel('Cosine distances') 
# 2. Fit a second order polynomial to the data in the scatter plot and overplot it. 
# INSERT YOUR CODE HERE
poly = PolynomialFeatures(degree = 2) 
X_poly = poly.fit_transform(all_euclidean_distances) 
  
poly.fit(X_poly, all_cosine_distances) 
lin2 = LinearRegression() 
lin2.fit(X_poly, all_cosine_distances) 
plt.plot(all_euclidean_distances, lin2.predict(poly.fit_transform(all_euclidean_distances)), color = 'red') 
  
plt.show() 






'''
Initialise a kmeans object  from scikit-lean package
'''
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=5, max_iter=3000,
                verbose=True, tol=0.000001, random_state=123456)


# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL

'''
Use the variable kmeans to perform clustering on the data stored in the variable all_termdoc
Hint: revise the practical session on Kmeans algorithm or check out the documentation from scikit-learn
for Kmeans algorithm.
'''
# INSERT YOUR CODE HERE
print(all_termdoc)
kmeans_cluster = kmeans.fit(all_termdoc)



# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
'''
Write your codes to print out the cluster centers.
'''
# INSERT YOUR CODE HERE
center = kmeans_cluster.cluster_centers_
print(center)





# YOU ARE REQUIRED TO INSERT YOUR COMMENT IN THIS CELL
# Explain below why visualising the clusters here is hard to do in this case.
# INSERT YOUR COMMENT HERE






# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
# 1. Plot bar charts for each of the three clusters, obtained from KMeans, 
# where each bar chart has 20 strongest words sorted by their presence strength.
# INSERT YOUR CODE HERE

height = center[0]
y_pos = np.arange(len(height))
plt.bar(y_pos, height)
plt.show()


height = center[1]
y_pos = np.arange(len(height))
plt.bar(y_pos, height)
plt.show()


height = center[2]
y_pos = np.arange(len(height))
plt.bar(y_pos, height)
plt.show()




# YOU ARE REQUIRED TO INSERT YOUR COMMENT IN THIS CELL
# Explain the bar charts from the point of view of chosen keywords, English grammar 
# and our text preprocessing routine.
# INSERT YOUR COMMENT HERE
#






# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
'''
Write your codes to print out the first **200** cluster labels assigned to the first 200 tweets.
'''
# INSERT YOUR CODE HERE
firsttweets200 = all_termdoc[0:200]
labels = kmeans_cluster.predict(firsttweets200)
print(len(all_termdoc))
print(len(firsttweets200))
print(labels)






# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL

'''
Write your code to obtain the labels of tweets for each keyword
and store the labels of the first keyword in ***k1_labels***, 
the labels of the second keyword in ***k2_labels*** and
the labels of the third keyword in ***k3_labels***.
'''
# INSERT YOUR CODE HERE
k1_indx = len(k1_termdoc)
k2_indx = len(k2_termdoc)
k3_indx = len(k3_termdoc)


# print(len(all_tweets_processed),len(all_tweets_processed[0]),len(all_tweets_processed[1]), len(k1_tweets_processed), len(k2_tweets_processed), len(k3_tweets_processed))
# print(k1_indx,k1_indy,k2_indx,k2_indy,k3_indx,k3_indy, len(all_termdoc),len(all_termdoc[0]))
# k1_ptr = np.concatenate((k1_termdoc, np.zeros([k1_indx, len(all_termdoc[0])-k1_indy])), axis=1)
# k2_ptr = np.concatenate((k2_termdoc, np.zeros([k2_indx, len(all_termdoc[0])-k2_indy])), axis=1)
# k3_ptr = np.concatenate((k3_termdoc, np.zeros([k3_indx, len(all_termdoc[0])-k3_indy])), axis=1)
k1_ptr = all_termdoc[0:k1_indx]
k2_ptr = all_termdoc[k1_indx:(k1_indx+k2_indx)]
k3_ptr = all_termdoc[(k1_indx+k2_indx):(k1_indx+k2_indx+k3_indx)]

k1_labels = kmeans_cluster.predict(k1_ptr)
k2_labels = kmeans_cluster.predict(k2_ptr)
k3_labels = kmeans_cluster.predict(k3_ptr)
print(k1_labels)
print(k2_labels)
print(k3_labels)






# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
'''
Write your code to obtain the list of tweet indices of each keyword that are assigned to the first cluster.
Hint: you might want to use numpy.where function.
'''
# obtain the list of tweet indices of keyword k1 that are assigned to the first cluster
# means that to find tweet indices that have label 0 in k1_labels
k1_idx_label0 = [id for id in range(0, len(k1_labels)) if k1_labels[id] == 0]

# obtain the list of tweet indices of keyword k2 that are assigned to the first cluster
# means that to find tweet indices that have label 0 in k2_labels
k2_idx_label0 = [id for id in range(0, len(k2_labels)) if k2_labels[id] == 0]

# obtain the list of tweet indices of keyword k3 that are assigned to the first cluster
# means that to find tweet indices that have label 0 in k3_labels
k3_idx_label0 = [id for id in range(0, len(k3_labels)) if k3_labels[id] == 0]




# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL
'''
Plot a bar chart to visualise the number of tweets of each keyword that are assigned to the first cluster.
Hint: you need to plot a bar chart with three bars, 
each bar represents the number of tweets of each keyword that are assigned to the first cluster.
'''
# INSERT YOUR CODE HERE
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# words = [keywords[0] , keywords[1], keywords[2]]
# cluster0_nums = [len(k1_idx_label0), len(k2_idx_label0), len(k3_idx_label0)]
# ax.bar(words, cluster0_nums)
# plt.show()

height = [len(k1_idx_label0), len(k2_idx_label0), len(k3_idx_label0)]
bars = [keywords[0] , keywords[1], keywords[2]]
y_pos = np.arange(len(bars))
plt.bar(y_pos, height, color=[0, 0.4, 1])
plt.xticks(y_pos, bars)
plt.show()





# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL

# obtain the list of tweet indices of keyword k1 that are assigned to the second cluster
# means that to find tweet indices that have label 1 in k1_labels
k1_idx_label1 = [id for id in range(0, len(k1_labels)) if k1_labels[id] == 1]

# obtain the list of tweet indices of keyword k2 that are assigned to the second cluster
# means that to find tweet indices that have label 1 in k2_labels
k2_idx_label1 = [id for id in range(0, len(k2_labels)) if k2_labels[id] == 1]

# obtain the list of tweet indices of keyword k3 that are assigned to the second cluster
# means that to find tweet indices that have label 1 in k3_labels
k3_idx_label1 = [id for id in range(0, len(k3_labels)) if k3_labels[id] == 1]

# Plot a bar chart to visualise the number of tweets of each keyword that are assigned to the second cluster
# INSERT YOUR CODE HERE
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# words = [keywords[0] , keywords[1], keywords[2]]
# cluster1_nums = [len(k1_idx_label1), len(k2_idx_label1), len(k3_idx_label1)]
# ax.bar(words, cluster1_nums)
# plt.show()

height = [len(k1_idx_label1), len(k2_idx_label1), len(k3_idx_label1)]
bars = [keywords[0] , keywords[1], keywords[2]]
y_pos = np.arange(len(bars))
plt.bar(y_pos, height, color=[0, 0.4, 1])
plt.xticks(y_pos, bars)
plt.show()



# YOU ARE REQUIRED TO INSERT YOUR CODE IN THIS CELL

# obtain the list of tweet indices of keyword k1 that are assigned to the third cluster
# means that to find tweet indices that have label 2 in k1_labels
k1_idx_label2 = [id for id in range(0, len(k1_labels)) if k1_labels[id] == 2]

# obtain the list of tweet indices of keyword k2 that are assigned to the third cluster
# means that to find tweet indices that have label 2 in k2_labels
k2_idx_label2 = [id for id in range(0, len(k2_labels)) if k2_labels[id] == 2]

# obtain the list of tweet indices of keyword k3 that are assigned to the third cluster
# means that to find tweet indices that have label 2 in k3_labels
k3_idx_label2 = [id for id in range(0, len(k3_labels)) if k3_labels[id] == 2]

# Plot a bar chart to visualise the number of tweets of each keyword that are assigned to the third cluster
# INSERT YOUR CODE HERE

# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# words = [keywords[0] , keywords[1], keywords[2]]
# cluster2_nums = [len(k1_idx_label2), len(k2_idx_label2), len(k3_idx_label2)]
# ax.bar(words, cluster2_nums)
# plt.show()


height = [len(k1_idx_label2), len(k2_idx_label2), len(k3_idx_label2)]
bars = [keywords[0] , keywords[1], keywords[2]]
y_pos = np.arange(len(bars))
plt.bar(y_pos, height, color=[0, 0.4, 1])
plt.xticks(y_pos, bars)
plt.show()
