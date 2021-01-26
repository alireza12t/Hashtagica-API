import pickle
import re

import nltk
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib


nltk.download('stopwords')

app = Flask(__name__)


def preprocess(tweet):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    stopword_list = nltk.corpus.stopwords.words('english') + ['u', 'im', 'rt', 'ummm', 'b', 'dont', 'arent', 'ya',
                                                              'yall', 'isnt',
                                                              'cant', 'couldnt', 'wouldnt', 'wont', 'yr', 'aint',
                                                              'gonna', 'ur',
                                                              'didnt', 'r', 'wasnt', 'werent', 'might', 'maybe',
                                                              'doesnt', 'would', 'shes', 'hes', 'youre', 'omg', 'us',
                                                              'wow'] + stopwords.words('english')

    preposition = ['in', 'at', 'by', 'from', 'on', 'for', 'with', 'about', 'into', 'through', 'between', 'under',
                   'against', 'during', 'without', 'upon', 'toward', 'among', 'within', 'along', 'across', 'behind',
                   'near', 'beyond', 'using', 'throughout', 'despite', 'to', 'beside', 'plus', 'towards', 'concerning',
                   'onto', 'beneath', 'via']
    stopword_list += preposition
    ps = PorterStemmer()

    tweet = tweet.lower()
    # remove unwanted characters
    tweet = re.sub(r'(\\x[^\s][^\s])', "", tweet)
    # remove \n
    tweet = re.sub(r'\\n', ' ', tweet)
    # remove url and mentions
    tweet = re.sub(r"(?:\@|rt @|https?\://)\S+", " ", tweet)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # tokenize tweets and remove punctuations
    tokens = tokenizer.tokenize(tweet)
    # remove stopwords and stemming
    tokens = [ps.stem(word) for word in tokens if word not in stopword_list]
    # remove short words and numbers
    twt = ' '.join([token for token in tokens if token.isalpha() and len(token) > 2])
    return twt


def find_features(document, word_features):
    words = set(document)
    featureset = {}
    for w in word_features:
        featureset[w] = (w in words)
    return featureset


@app.route('/predict/', methods=['POST'])
def predict():
    features_file = open('word_features_file_2k.pickle', 'rb')
    features = pickle.load(features_file)

    model_file = open('multinumialNB_classifier_2k.pickle', 'rb')
    model = joblib.load(model_file)
    
    data = request.json["message"]
    processed = preprocess(data)
    prediction = model.classify(find_features(processed.split(), features))

    return jsonify({'prediction': [prediction]})


if __name__ == '__main__':
    

    app.run(debug=True)
