import pandas as pd
import numpy as np
import pickle

# Loading the dataset
df = pd.read_csv(r'C:\Users\admin\Desktop\Spam_SMS_Collection.txt', sep='\t', names=['label', 'message'])
# print(df.head())

# Importing essential libraries for performing Natural Language Processing on 'SMS Spam Collection' dataset
import nltk
import re

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Cleaning the messages
corpus = []
ps = PorterStemmer()

for i in range(0, df.shape[0]):
    # Cleaning special characters from the message
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df.message[i])

    # Convert the entire message into lower case
    message = message.lower()
    # print(message)

    # Tokenizing the review by words
    words = message.split()
    # print(words)

    # Removing the stopwords
    words = [word for word in words if word not in set(stopwords.words('english'))]
    # print(words)
#
#     # Stemming the words
    words = [ps.stem(word) for word in words]
    # print(words)

     # Joining the stem words
    message = ' '.join(words)
    # print(message)


    # Building a corpus of messages
    corpus.append(message)

# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
# print(X)
# Extracting dependent variables from the dataset
y = pd.get_dummies(df['label'])
# print(y)
y = y.iloc[:, 1].values

# Creating a pickle file for the CountVectorizer
# filename = r'C:\Users\admin\Desktop\cv-transform.pkl'
# pickle.dump(cv, open(filename, 'wb'))

# Model Building

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
# from sklearn.naive_bayes import MultinomialNB
# classifier = MultinomialNB(alpha=0.3)
# classifier.fit(X_train, y_train)

# Creating a pickle file for the Multinomial Naive Bayes model
# filename = r'C:\Users\admin\Desktop\spam-sms-mnb-model.pkl'
# pickle.dump(classifier, open(filename, 'wb'))