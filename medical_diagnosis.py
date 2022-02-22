
# Medical Diagnosis App which takes symptoms list as input and returns the list of top 10 diseases with their probabilities using the Decision Tree Classifier
# PywebIO is used to host the web interface

# all necessary imports
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from statistics import mean
from nltk.corpus import wordnet 
import requests
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")
import requests
from bs4 import BeautifulSoup
import streamlit as st

# ignore warnings generated due to usage of old version of tensorflow
warnings.simplefilter("ignore")



def predict_diseases(userSymptoms):
    # use the dataset scraped from Wikipedia and NHP to predict the disease
    # load the dataset
    combinedDF = pd.read_csv('Diagnosis-App\dataset\diseaseSymptomComb.csv')
    # remove the rows with missing values
    combinedDF = combinedDF.dropna()
    normalizedDF = pd.read_csv('Diagnosis-App\dataset\diseaseSymptomNormal.csv')
    # remove the rows with missing values
    normalizedDF = normalizedDF.dropna()

    # Features and Labels
    X = combinedDF.iloc[:, 1:]
    Y = combinedDF.iloc[:, 0:1]

    # Get the list of total symptoms from X which are the columns of the dataframe
    total_symptoms = list(X.columns)

    # Get the list of unique diseases from Y
    # diseases = Y.unique().tolist()

    # Get the list of unique symptoms from X
    # unique_symptoms = X.columns.unique().tolist()

    # Initialise sample_x with 0s
    sample_x = []
    for i in range(0, len(total_symptoms)):
        sample_x.append(0)
    
    # Iterate over all the symptoms in userSymptoms and set the corresponding index of sample_x to 1
    for symptom in userSymptoms:
        index = total_symptoms.index(symptom)
        sample_x[index] = 1

    # Predict disease using Decision Tree Classifier
    decisionTree = DecisionTreeClassifier()
    decisionTree = decisionTree.fit(X, Y)
    
    #define the scoresDT to store the cross validation scores
    scoresDT = cross_val_score(decisionTree, X, Y, cv=5)
    # get the prediction probabilities for the sample_x
    predictionDT = decisionTree.predict_proba([sample_x])


    disease_list = list(set(Y['label_dis'])) # get the list of all the diseases from the dataset
    # sort the disease list
    disease_list.sort()
    top_10_diseases = predictionDT[0].argsort()[-10:][::-1] # get the top 10 diseases

    # create a dictionary to hold the top 10 diseases with their probabilities
    top_10_diseases_dict = {}

    # Iterate over the prediction probabilities and add the top 10 diseases to the dictionary
    for index, i in enumerate(top_10_diseases):
        # create a set of matched_symptoms
        matched_symptoms = set()
        # get the disease name
        # disease = disease_list[i]
        # get the disease index
        disease_index = normalizedDF.loc[normalizedDF['label_dis'] == disease_list[i]].values.tolist()
        disease_index[0].pop(0)


        for index, j in enumerate(disease_index[0]):
            if j!=0:
                matched_symptoms.add(total_symptoms[index])
            
        # find the probability of the disease
        probability = (len(matched_symptoms.intersection(set(userSymptoms)))+1)/(len(set(userSymptoms))+1) # add 1 to avoid division by zero
        probability = probability*mean(scoresDT) # multiply the probability with the mean of the cross validation scores
        top_10_diseases_dict[i] = probability

    j = 0
    # create a dictionary to store the top 10 index mappings for diseases and their probabilities
    top_10_diseases_index_dict = {}
    # for key, value in top_10_diseases_dict.items():
    #     top_10_diseases_index_dict[j] = [key, value]
    #     j += 1
    
    # sort the dictionary based on the probabilities
    sorted_top_10_diseases = dict(sorted(top_10_diseases_dict.items(), key=lambda x: x[1], reverse=True))

    # Now, store the disease name with its probability in final_diseases_dict
    final_diseases_dict = []
    cnt = 0
    for i in sorted_top_10_diseases:
        probability = sorted_top_10_diseases[i]*100
        final_diseases_dict.append([disease_list[cnt], round(probability,2)])
        top_10_diseases_index_dict[j] = i
        j += 1
        cnt += 1


    # return the top 10 diseases and their probabilities
    return final_diseases_dict


combDF = pd.read_csv('Diagnosis-App\dataset\diseaseSymptomComb.csv')
# remove the rows with missing values
combDF = combDF.dropna()
normDF = pd.read_csv('Diagnosis-App\dataset\diseaseSymptomNormal.csv')
# remove the rows with missing values
normDF = normDF.dropna()

# Features and Labels
Xa = combDF.iloc[:, 1:]
Ya = combDF.iloc[:, 0:1]

# Get the list of total symptoms from X which are the columns of the dataframe
symptoms_list = list(Xa.columns)


st.title('Medical Diagnosis App')
st.subheader('Predicting the disease based on the symptoms')
st.text('Please enter the symptoms you are experiencing in the sidebar.')

st.sidebar.header('User Input Parameters')
st.sidebar.subheader('Select your symptoms')
symptoms = st.sidebar.multiselect('Select', symptoms_list)

# When the user clicks the submit button
if st.sidebar.button('Submit'):
    
    # show users the selected symptoms using a checkbox
    st.info('You selected the following symptoms:')
    st.write(symptoms)
    
    # Calling the predict_disease function when the button is pressed
    ans = predict_diseases(symptoms)
    
    st.subheader('Disease Prediction by the Decision Tree Classifier')
    # st.success(disease1[0] + f" with a confidence of {disease1[1][disease1[0]]*100:.2f}%" )
    st.write('**The top 10 results from the model along with their corresponding chances of occurrence(in %) are:**')
    st.write(ans)

