# Medical-Diagnosis-App
An app hosted using Streamlit and uses the Decision Tree Classifier to suggest top 10 diseases based on the symptoms entered by the user.

**This project has been done as a part of the Assignment in the course Data Ananlytics and Visualisation.**


**Find the explanation and brief report of the project in [this document](https://github.com/Anupam0401/Medical-Diagnosis-App/blob/master/Medical%20Diagnosis%20App.pdf).**
---
# Run the App on your local machine
* First, install the required libraries to run the code.
```
pip install requirements.txt
```
* Now, run the *medical_diagnosis.py* file using the command
```
python3 medical_diagnosis.py
```

### If you would like to scrape data or re-train the dataset, follow the following steps:
```
python3 -m pip install googlesearch-python
```
```
pip install -U nltk
```
* Now, run the *preprocess.py* to scrape data and store to the csv file.
