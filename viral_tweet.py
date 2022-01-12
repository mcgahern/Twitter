import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# Importing and Inspecting the data file
all_tweets = pd.read_json("random_tweets.json", lines=True)

#print(len(all_tweets))
#print(all_tweets.columns)


#Print the user here and the user's location here.
print(all_tweets.loc[0]['user'])
print(all_tweets.loc[0]['user']['location'])

# Defining a Viral Tweet
#print(all_tweets['retweet_count'].median())
all_tweets['is_viral'] = np.where(all_tweets['retweet_count'] > 13,1,0)
print(all_tweets['is_viral'].value_counts())


# Making Features
all_tweets['tweet_length'] = all_tweets.apply(
    lambda tweet: len(tweet['text']), axis=1)
all_tweets['followers_count'] = all_tweets.apply(
    lambda tweet: tweet['user']['followers_count'], axis=1)
all_tweets['friends_count'] = all_tweets.apply(
    lambda tweet: tweet['user']['friends_count'], axis=1)
all_tweets['hashtags'] = all_tweets.apply(
    lambda tweet: tweet['text'].count('#'), axis=1)
all_tweets['word_count'] = all_tweets.apply(
    lambda tweet: len(tweet['text'].split()), axis=1)


# Normalizing the Data
labels = all_tweets['is_viral']
data = all_tweets[['tweet_length', 'followers_count',
                   'friends_count', 'hashtags', 'word_count']]
scaled_data = scale(data)
print(scaled_data[0])

# Creating the Training and Test Set
train_data, test_data, train_labels, test_labels = \
    train_test_split(data, labels, test_size=0.2, random_state=1)


# Choosing K + Using the Classifier
scores = []
for k in range(1,200):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_data, train_labels)
    scores.append(classifier.score(test_data, test_labels))
plt.plot(range(1,200), scores)
plt.xlabel('n_neighbors value')
plt.ylabel('Score (%)')
plt.show()