import numpy as np
import pandas as pd 
import re
import nltk 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib



# Read Data:

# ticket_data = pd.read_csv('test_hypo.csv'): Reads a CSV file named 'test_hypo.csv' into a Pandas DataFrame called ticket_data. 
# This file presumably contains the training data for your text classification task.
# Text Preprocessing:

# stemming function: This function preprocesses text data by converting it to lowercase, removing non-alphabetic characters, 
# tokenizing it into words, removing stop words, and applying stemming to reduce words to their root form.
#  This function is applied to the 'content' column in the training data but is currently commented out.

# Feature Engineering:
# The script combines several columns (e.g., 'title', 'user_email', 'description', etc.) into a single 'content' column in the training data. 
# This step is necessary to create a text representation of the data that can be used for modeling.
# Text Vectorization:

# The script uses the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to convert the text data into numerical features.
#  It fits the vectorizer on the training data (X) and then transforms both the training and testing data into TF-IDF representations. 
# This step is essential for turning text data into a format that machine learning models can work with.

# Model Training:
# It splits the data into training and testing sets using train_test_split.
# It initializes a logistic regression model.
# It trains the model on the training data using the TF-IDF representations of text data (X_train).

# Model Evaluation:
# It calculates and prints the accuracy score of the model on both the training and testing data. Accuracy is a measure of how well the model classifies text data.
# Model Saving:
# It saves the trained logistic regression model to a file named 'trained_model_ver2.pkl' using joblib. This allows you to reuse the trained model for making predictions without retraining it.
# Prediction Function:
# The make_prediction function takes a CSV file path as input, reads the data from that file, applies the same preprocessing steps as the training data, makes predictions using the trained model, adds the predictions as a new column ('Predictions') to the data, saves the modified data to a new CSV file with a name like 'outputmika.csv', and returns the path to the output CSV file.


# print(stopwords.words('english'))

ticket_data = pd.read_csv('mikaNewDataToTrain.csv')

port_stem = PorterStemmer()

def stemming(content):
    if isinstance(content, str):
        stemmed_content = content
        stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
        stemmed_content = ' '.join(stemmed_content)
        return stemmed_content
    else:
        return ""

def make_prediction(csv_file_path, save_model_path='trained_ver4.pkl'):
    dataframe = pd.read_csv(csv_file_path)

    # Execute the prediction
    X = dataframe['content'].values

    # Load the saved model and vectorizer
    saved_data = joblib.load(save_model_path)
    loaded_model = saved_data['model']
    loaded_vectorizer = saved_data['vectorizer']

    # Transform the text data using the loaded vectorizer
    X = loaded_vectorizer.transform(X)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(X)

    dataframe['Predictions'] = predictions

    output_csv_path = 'output' + csv_file_path

    dataframe.to_csv(output_csv_path, index=False)

    print(f'Saved predictions to {output_csv_path}')

    return output_csv_path

make_prediction('translated.csv')






# ticket_data['content'] = ticket_data['content'].apply(stemming)   ### stemming and shit ###
# # Save the modified DataFrame to a new CSV file

# #####################################
# X= ticket_data['content'].values
# Y= ticket_data['priority'].values


# vectorizer = TfidfVectorizer()
# vectorizer.fit(X)

# X = vectorizer.transform(X)

# #splitting the dataset to traning & test data
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify = Y, random_state=2)

# #training the Model: Logistic Regression

# model = LogisticRegression()

# model.fit(X_train, Y_train)

# #Eveluation
# #accuracy score on the training date
# X_train_prediction = model.predict(X_train)
# training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
# print(f'accuracy score of the training data {training_data_accuracy}')

# #accuracy score on the test data
# X_test_prediction = model.predict(X_test)
# test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
# print(f'accuracy score of the test data {test_data_accuracy}')

# # Save both the model and vectorizer to a single file
# joblib.dump({'model': model, 'vectorizer': vectorizer}, 'trained_ver5.pkl')


# #Making a Predictive System

