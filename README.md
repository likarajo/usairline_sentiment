# US Airlines Sentiment

## Goal
Based on tweets about six US airlines, we predict whether a tweet contains positive, negative, or neutral sentiment about the airline. We categorize the text string into predefined categories using supervised machine learning.

## Process
* Perform exploratory data analysis look for trends in the dataset. 
* Preprocess data to convert text to numeric data that can be used by a machine learning algorithm. 
* Train and test model
* Evaluate the model

## Python Libraries
* jupyter
* numpy
* pandas
* re
* nltk
* matplotlib 
* seaborn 
* sklearn
* pickle-mixing

`pip install -r requirements.txt`

### Stopwords
```py
import nltk
nltk.download('stopwords')
```

# Data Analysis
* Start Jupyter notebook *data_analysis.ipynb*
* The 11th column (index 10) contains the tweet text.
* The 2nd column (index 1) contains the sentiment of the tweet.
* Number of Tweets for each airline: United Airline 26%, followed by US Airways 20% etc.
* Distribution of sentiment across all airlines: negative 63%, neutral 21%, positive 16%.
* Distribution of sentiment for each individual airline: For all the airlines, the majority of the tweets are negative, followed by neutral and positive tweets.
* Average confidence level for the tweets for each sentiment category: confidence level for negative tweets is higher compared to positive and neutral tweets.

---

# Model Creation
* Start Jupyter notebook *data_cleaning.ipynb*

## Data Cleaning

* Divide the dataset into features and labels sets
    * features = text = 11th column (index 10)
    * labels = airline_sentiment = 2nd column (index 1)
* Preprocess features
    * Replace special charcters with a space
    * Replace all single charcaters with a space
    * Remove single characters from the start with a space
    * Substitute multiple spaces with a single space
    * Remove prefixed 'b' present for bytes format strings
    * Convert to lower case

## Representing Text in Numeric form
Statistical algorithms use mathematics to train machine learning models. 
To make statistical algorithms work with text, we convert text to numbers. 
To do so, three main approaches exist i.e. ***Bag of Words***, ***TF-IDF*** and ***Word2Vec***.

### Bag of Words
* Create a vocabulary set of all the unique words in the given documents
* Convert each document into a feature vector using the vocabulary set
    * Length of the feature vector is equal to length of the vocabulary set
    * The frequency of the word in the document will replace the actual word in the vocabulary. If a word in the vocabulary is not found in the corresponding document, the document feature vector will have zero in that place. 
* Each word has the same weight  
Example:  
* Doc1 = "I like to play football"
* Doc2 = "It is a good game"
* Doc3 = "I prefer football over rugby"
* vocab = [I, like, to, play, football, It, is, a, good, game, prefer, over, rugby]
* Doc1_featureVector = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
* Doc2_featureVector = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
* Doc3_featureVector = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]

### TF-IDF
* **Term frequency** and **Inverse Document frequency**
* Words that occur less in all the documents and more in individual document contribute more towards classification. 
* **TF  = *Frequency of a word in the document* / *Total words in the document***
* **IDF = Log( *Total number of docs* / *Number of docs containing the word* )**

Python's Scikit-Learn library contains the ***TfidfVectorizer*** class that can be used to convert text features into TF-IDF feature vectors.

## Splitting data into Training and Test sets
The training set will be used to train the algorithm while the test set will be used to evaluate the performance of the machine learning model.

## Training the model
Train the model using training data set.
*  Random Forest algorithm is used to train the model as it is able to act upon non-normalized data.

## Making predictions
Make predictions on the trained model using test features data set.

## Evaluating the model
Evaluate the performance of the machine learning model against test labels using classification metrics
* Confusion matrix
* Classification report
* Accuracy score

## More classifiers
More models can be trained to get better accuracy.  
* Logistic regression
* SVM
* KNN 

## Save and/or load the model
The fitted model can be saved and loaded later to use for predictions on new feature data.
