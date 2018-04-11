The aim of this project is to perform Sentiment Analysis of Yelp ‘s Ratings Based on Text Reviews and compare accuracies of different classifiers. 
Data files used here is yel_dataset.csv that is obtained from https://www.yelp.com/dataset/challenge in json format and converted into csv.

Files enclosed and instructions to execute them:

Project.py - The main file to run.
yelp_dataset.csv - The dataset file which contains all the reveiws and star ratings.
Here, 1 represents negative review and 5 represents the positive review.

Steps to run:
Place the yelp_dataset.csv file in the same location as Project.py file.
For running the program please run the below command preceded by the location where this file is located in command prompt or terminator.
location of file > python Project.py

Packages to be installed:
1) Pandas for reading csv file --- command: pip install Pandas 
2) NLTK package for stopwords removal --- command: pip install NLTK
3) Seaborn and matplotlib for exploring and visualizing data --- command: pip install Seaborn, pip install matplotlib
4) sklearn --- command: pip install sklearn (This is used for using Multinomial naive Bayes, KNN and SVM classifiers)


