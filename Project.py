"""
@Authors: Neel Dave
          Rangita Rajakumar
          Syed Abrar Ahmed
"""

import pandas as pa
import matplotlib.pyplot as mp
import seaborn as sb

import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn import metrics, svm
from sklearn.neighbors import KNeighborsClassifier

# Reads the yelp_dataset file and set it as a dataframe
yelp_data = pa.read_csv('yelp_dataset.csv', encoding='latin-1')

def basic_info():
    """
    Gives the basic information about the data
    like number of rows, columns, type data etc..
    """

    print(yelp_data.shape)
    print(yelp_data.head())
    print(yelp_data.info())
    print(yelp_data.describe())

basic_info()

""" A new column text length is created to store the number of characters in each review """

yelp_data['text length'] = yelp_data['text'].apply(len)
print(yelp_data.head())

def explore_data():
    """
    Visualizes data by plotting few graphs
    with the use of seaborn libraries
    """

    g = sb.FacetGrid(data=yelp_data, col='stars')
    g.map(mp.hist, 'text length', bins=50, color='#2a286e')
    mp.savefig('stars_vs_textlength', dpi=200)

    sb.boxplot(x='stars', y='text length', data=yelp_data, palette=sb.cubehelix_palette(9, start=.4, rot=-.70, reverse=True, light=0.85, dark=0.25))
    mp.savefig('stars_textlen_box', dpi=200)

    sb.countplot(x='stars', data=yelp_data, palette=sb.cubehelix_palette(9, start=.4, rot=-.70, reverse=True, light=0.85, dark=0.25))

    # Use groupby to get the mean values of the numerical columns
    stars_mean = yelp_data.groupby('stars').mean()
    print(stars_mean)

    # Use the corr() method on that groupby dataframe to produce this dataframe
    print(stars_mean.corr())

    cu_map = sb.cubehelix_palette(9, start=.4, rot=-.70, as_cmap=True, reverse=False, light=0.85, dark=0.35)
    sb.heatmap(data=stars_mean.corr(), annot=True, cmap=cu_map)
    mp.savefig('heat_map', dpi=200)

explore_data()

def stars_1_5():
    """
    Creates a dataframe that contains the columns
    of yelp_data for only 1 or 5 star reviews as we are
    focused on good and bad reviews
    """

    yelp_data_1_5 = yelp_data[(yelp_data['stars'] == 1) | (yelp_data['stars'] == 5)]

    print(yelp_data_1_5.head())
    print(yelp_data_1_5.shape)
    print(yelp_data_1_5['stars'].unique())

    """Creates two objects text_col and stars_col. text_col will be the 'text' column 
    of yelp_data_1_5 and stars_col will be the 'stars' column of yelp_data_1_5."""

    text_col = yelp_data_1_5['text']
    stars_col = yelp_data_1_5['stars']

    return text_col, stars_col

x_text, y_stars = stars_1_5()

def text_process(text):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuations
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """

    no_punc = []
    for char in text:
        if char not in string.punctuation:
            no_punc.append(char)

    no_punc = ''.join(no_punc)

    word_list = []
    for word in no_punc.split():
        if word.lower() not in stopwords.words("english"):
            word_list.append(word)

    return word_list

def vectorization():
    """
    Converts the text collection into a matrix of tokens
    and transforms the dataframe into a sparse matrix
    """
    bow_transformer = CountVectorizer(analyzer=text_process).fit(x_text)
    print(len(bow_transformer.vocabulary_))

    x_trans = bow_transformer.transform(x_text)
    print(x_trans)

    print('Shape of Sparse Matrix: ', x_trans.shape)
    print('Amount of Non-Zero occurences: ', x_trans.nnz)

    sparsity = (100.0 * x_trans.nnz / (x_trans.shape[0] * x_trans.shape[1]))
    print('sparsity: {}'.format(sparsity))

    return x_trans

x = vectorization()

x_train, x_test, y_train, y_test = train_test_split(x, y_stars, test_size=0.3, random_state=101)

def training_MNB():
    """
    Multinomial Naive Bayes is a specialised version of Naive Bayes designed more for text documents.
    Multinomial Naive Bayes model is built and fit it to our training set (x_train and y_train).
    """

    nb = MultinomialNB()
    nb.fit(x_train, y_train)

    return nb

mnb = training_MNB()

def test_MNB():
    """
    Tests the MNB trained model and prints the stats
    """
    pred = mnb.predict(x_test)
    accuracy_MNB = metrics.accuracy_score(y_test, pred)*100
    f1_MNB = f1_score(y_test, pred, average="binary")*100

    print(accuracy_MNB)
    print(f1_MNB)

    print(confusion_matrix(y_test, pred))
    print('\n')
    print(classification_report(y_test, pred))
    return accuracy_MNB, f1_MNB

MNB_accuracy, MNB_f1_score = test_MNB()

def training_KNN():
    """
    k-nearest neighbors model is built and fit it to our training set (x_train and y_train).
    """

    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(x_train, y_train)

    return neigh

knn = training_KNN()

def test_KNN():
    """
    Tests the KNN trained model and prints the stats
    """
    pred = knn.predict(x_test)
    accuracy_knn = metrics.accuracy_score(y_test, pred)*100
    f1_knn = f1_score(y_test, pred, average="binary")*100

    print(accuracy_knn)
    print(f1_knn)

    print(confusion_matrix(y_test, pred))
    print('\n')
    print(classification_report(y_test, pred))
    return accuracy_knn, f1_knn

KNN_accuracy, KNN_f1_score = test_KNN()

def training_svm():
    """
    Support Vector Machine model is built and fit it to our training set (x_train and y_train).
    """

    s_v_m = svm.SVC()
    s_v_m.fit(x_train, y_train)

    return s_v_m

su_vm = training_svm()

def test_svm():
    """
    Tests the SVM trained model and prints the stats
    """
    pred = su_vm.predict(x_test)
    accuracy_svm = metrics.accuracy_score(y_test, pred)*100
    f1_svm = f1_score(y_test, pred, average="binary")*100

    print(accuracy_svm)
    print(f1_svm)

    print(confusion_matrix(y_test, pred))
    print('\n')
    print(classification_report(y_test, pred))
    return accuracy_svm, f1_svm

svm_accuracy, svm_f1_score = test_svm()

def print_accuracy():
    print("The accuracy using Multinomial Naive Bayes classifier is: ", MNB_accuracy)
    print("The accuracy using k-nearest neighbor classifier is: ", KNN_accuracy)
    print("The accuracy using support vector machine classifier is: ", svm_accuracy)

    if MNB_accuracy > KNN_accuracy and MNB_accuracy > svm_accuracy:
        print("\nThus, Sentiment analysis using Multinomial Naive Bayes classifier is more accurate.")
    elif KNN_accuracy > MNB_accuracy and KNN_accuracy > svm_accuracy:
        print("\nThus, Sentiment analysis using k-nearest neighbor classifier is more accurate.")
    else:
        print("\nThus, Sentiment analysis using support vector machine classifier is more accurate.")

print_accuracy()

