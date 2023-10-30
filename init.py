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


# print(stopwords.words('english'))

ticket_data = pd.read_csv('final2.csv')

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
    

def make_prediction(csv_file_path): 
    dataframe = pd.read_csv(csv_file_path)

    columns_to_combine = [
        'title', 'user_email', 'user_team', 'description',
        'product_line', 'doi_tuong_anh_huong', 'platform_anh_huong'
    ]
    # Create the 'content' column by concatenating the selected columns
    dataframe['content'] = dataframe[columns_to_combine].apply(lambda row: ' '.join(row), axis=1)

    #execute the prediction
    X = dataframe['content'].values

    loaded_model = joblib.load('trained_model.pkl')

    predictions = loaded_model.predict(X)

    dataframe['Predictions'] = predictions

    output_csv_path = 'output' + csv_file_path

    dataframe.to_csv(output_csv_path, index=False)

    print (f'save to {output_csv_path}')

    return output_csv_path

# # ticket_data['content'] = ticket_data['content'].apply(stemming)   ### stemming and shit ###
# # Save the modified DataFrame to a new CSV file


X= ticket_data['content'].values
Y= ticket_data['priority'].values


vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

#splitting the dataset to traning & test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify = Y, random_state=2)

#training the Model: Logistic Regression

model = LogisticRegression()

model.fit(X_train, Y_train)

#Eveluation
#accuracy score on the training date
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(f'accuracy score of the training data {training_data_accuracy}')

#accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f'accuracy score of the test data {test_data_accuracy}')

joblib.dump(model, 'trained_model.pkl')

    

#Making a Predictive System

