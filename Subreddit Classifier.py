
# coding: utf-8

# In[ ]:


import numpy as np
import re
import json

from sklearn.feature_extraction import stop_words
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

import gensim.models.keyedvectors as word2vec

def getVector(w):
    global model
    if w in model:
        return model[w]
    else:
        return np.zeros(300)


def cos_sim(w1,w2):
    return np.dot(getVector(w1),getVector(w2)) / ( np.linalg.norm(getVector(w1)) * np.linalg.norm(getVector(w2)))


def extract_from_corpus(corpus, capture_words):
    # create bag of words and sum up word occurrences to find most frequent words in the given corpus
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    
    # only keep words that have a count >= 60, and have cosine similarity >= 0.28 from words relevant to the context of the subreddit
    # (these words are themselves common in the subreddit, and were chosen based on observations made manually by sifting through the words common in the corpus)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items() if ((sum_words[0,idx] >= 60) and (cos_sim(word,capture_words[0]) >= 0.28 or cos_sim(word,capture_words[1]) >= 0.28))]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    relevant_common_words = [w[0] for w in words_freq]
    
    return relevant_common_words


def classifySubreddit_train(trainFile):
    
    # list to store the data from the training file
    data = []
    
    # collect data from json file
    with open(trainFile, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    f.close()
    
    # corpi to store the corpus of each subreddit
    corpus_nerf = []
    corpus_uke = []
    corpus_navy = []
    corpus_mario = []
    
    # comments stores the processed text of all the comments
    comments = []
    # comment_vecs stores vectors of all the comments
    comment_vecs = []
    # subreddits stores the subreddit of each comment
    subreddits = []
    
    # used to join text later
    space = " "
    
    global stops
    global zero_arr
    
    for d in data:
        # remove punctuation from each comment, make it lowercase
        comment = d['body']
        comment = comment.lower()
        comment = re.sub(r'[^a-zA-Z0-9 ]', ' ', comment)
        comment = re.sub(r' +',' ', comment)
        
        # split comment into tokens/words and keep only the tokens that are not stopwords and have a vector in the word2vec model
        tokens = comment.split()
        tokens = [w for w in tokens if (w not in stops) and not (np.array_equal(w,zero_arr))]
        
        # join the tokens back to form the processed text of the comment
        text = ""
        text = space.join(tokens)
        
        # append the processed text to its respective corpus, and append the subreddit name to the list of subreddits
        if d['subreddit'] == 'Nerf':
            corpus_nerf.append(text)
            subreddits.append(0)
        elif d['subreddit'] == 'ukulele':
            corpus_uke.append(text)
            subreddits.append(1)
        elif d['subreddit'] == 'newToTheNavy':
            corpus_navy.append(text)
            subreddits.append(2)
        else:
            corpus_mario.append(text)
            subreddits.append(3)
        
        # append the text to the list of all comments
        comments.append(text)
    
    # get a list of words common in each corpus that are relevant to the corpus' topic
    relevant_common_words_nerf = extract_from_corpus(corpus_nerf, ['nerf','play'])
    relevant_common_words_uke = extract_from_corpus(corpus_nerf, ['ukulele','music'])
    relevant_common_words_navy = extract_from_corpus(corpus_nerf, ['navy','job'])
    relevant_common_words_mario = extract_from_corpus(corpus_nerf, ['mario','mario'])
    
    # combine the lists
    all_words = relevant_common_words_nerf + relevant_common_words_uke + relevant_common_words_navy + relevant_common_words_mario
    
    # remove the words that are common in more than one corpus
    global all_uni_words
    all_uni_words = [w for w in all_words if all_words.count(w) == 1]
    
    # iterate through the processed comments, and generate a list where each element corresponds to one of the common words across all corpi
    # add 1 if the word is present in the text, 0 if it is not
    for c in comments:
        vec_comment = np.zeros(300)
        common_words = []
        
        tokens = c.split()
        
        for w in tokens:
            vec_comment += getVector(w)
        
        for w in all_uni_words:
            if w in tokens:
                common_words.append(1)
            else:
                common_words.append(0)
        
        # convert this list of common words presence indicator to an array
        common_words = np.array(common_words)
        # append this array to the original word2vec array of that comment
        vec_comment = np.append(vec_comment,common_words)

        # add this final comment array to the list of all comment arrays
        comment_vecs.append(vec_comment)
    
   # convert list of arrays of comments to an array of arrays of comments for training
    X = np.stack(comment_vecs)
    # convert list of subreddits to an array of subreddits
    y = np.array(subreddits)

    # define the model
    global logreg
    logreg = LogisticRegression()

    # fit the model onto the training data
    logreg.fit(X, y)


def classifySubreddit_test(comment):
    
    global stops
    global zero_arr

    # vector to store the sum of vectors of all words in the comment
    vec_comment = np.zeros(300)

    # process the comment: make it lowercase, remove punctuation and stopwords, remove words not represented in word2vec model
    comment = comment.lower()

    comment = re.sub(r'[^a-zA-Z0-9 ]', ' ', comment)
    comment = re.sub(r' +',' ', comment)
        
    tokens = comment.split()
    tokens = [w for w in tokens if (w not in stops) and not (np.array_equal(w,zero_arr))]

    # sum up the vectors of all words in the processed text
    for w in tokens:
        vec_comment += getVector(w)
    
    common_words = []
    
    global all_uni_words

    # find which of the common words from the subreddits are present, append 1 for present, 0 for not present
    for w in all_uni_words:
        if w in tokens:
            common_words.append(1)
        else:
            common_words.append(0)

    # append array representing common words to the original word2vec array of the sentence
    vec_comment = np.append(vec_comment,np.array(common_words))
    
    global logreg
    # predict subreddit using the model
    result = logreg.predict([vec_comment])

    # return the name of the subreddit
    if result[0] == 0:
        return "Nerf"
    elif result[0] == 1:
        return "ukulele"
    elif result[0] == 2:
        return "newToTheNavy"
    else:
        return "MarioMaker"


# In[ ]:


print("Retrieving word2vec vectors. This may take a few minutes...")
model = word2vec.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary=True)


# In[ ]:


# global variables to be used in subreddit classification
stops = stop_words.ENGLISH_STOP_WORDS
zero_arr = np.zeros(300)
logreg = 0
all_uni_words = []

# training file and testing file paths
trainFile = "redditComments_train.jsonlist"
testFile = "redditComments_test.jsonlist"


# In[ ]:


# call training function
classifySubreddit_train(trainFile)


# In[ ]:


# list to store the data from the testing file
test_data = []
    
# collect data from json file
with open(testFile, 'r') as f:
    for line in f:
        test_data.append(json.loads(line))

f.close()

# variable to store how many test cases we got right
correct = 0

# for loop to test the model
for d in test_data:
    if d['subreddit'] == classifySubreddit_test(d['body']):
        correct+=1

# calculate and print accuracy
accuracy = correct/len(test_data)

print("Accuracy on test set: ", accuracy*100)

