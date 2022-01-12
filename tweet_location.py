import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Importing and Investigating Data sets
new_york_tweets = pd.read_json("new_york.json", lines=True)
#print(len(new_york_tweets))
#print(new_york_tweets.columns)
#print(new_york_tweets.loc[12]["text"])

paris_tweets = pd.read_json('paris.json', lines=True)
#print(len(paris_tweets))

london_tweets = pd.read_json('london.json', lines=True)
#print(len(london_tweets))

# Naive Bayes Classifier
new_york_text = new_york_tweets["text"].tolist()
paris_text = paris_tweets['text'].tolist()
london_text = london_tweets['text'].tolist()

all_tweets = new_york_text + paris_text + london_text
labels = [0] * len(new_york_text) + [1] * len(london_text) + [2] * len(paris_text)

# Make a Training and Test Set
train_data, test_data, train_labels, test_labels = train_test_split(
    all_tweets, labels, test_size=0.2, random_state=1)

# Making Count Vectors
counter = CountVectorizer()
counter.fit(train_data)

train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)

#print(all_tweets[3])
#print(train_counts[3])
#print(test_counts[3])

# Test and Train Naive Bayers
classifier = MultinomialNB()
classifier.fit(train_counts, train_labels)
predictions = classifier.predict(test_counts)

# Evaluating Your Model
print(accuracy_score(test_labels, predictions))
print(confusion_matrix(test_labels, predictions))

# Test Your Own Tweet
tweet = 'Your auld lady has worms'
tweet_counts2 = counter.transform([tweet])
print(classifier.predict(tweet_counts2))
