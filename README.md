# Resume_screening-project

To help you structure your project based on the Resume Screening App you built, I will provide an overview, features, model architecture, dataset, training, and results similar to your image captioning project structure.

# Overview

This project implements a Resume Screening App that classifies resumes into predefined job categories using machine learning models. It combines Natural Language Processing (NLP) and machine learning techniques to predict job roles based on resume content.

# Table of Contents
* Overview
* Features
* Dependencies
* Model Architecture
* Dataset
* Training
* Results

# Features
* Resume Classification: The model predicts job categories based on the uploaded resume text.
* Machine Learning: Uses a pre-trained TF-IDF vectorizer and classification model to process and classify resumes.
* Streamlit Web App: Provides an easy-to-use interface for uploading resumes and viewing predicted job categories.
# Dependencies
* Python 3.x
* Streamlit
* Scikit-learn
* NLTK
* Pickle (for loading pre-trained models)
* 
# Model Architecture
The resume screening model consists of:

* Text Preprocessing: Using NLTK for tokenization, stopword removal, and cleaning the resume text.
* Feature Extraction: TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert resume text into numerical features.
* Classification Model: A pre-trained classifier (like Logistic Regression or SVM) is used to predict job categories based on extracted features.
  
# Dataset
The model was trained on a dataset of resumes provided by Gaurav Dutt. The dataset contains resumes and their corresponding job categories, which serve as labels for training the model.

# Training
Training involves several steps outlined in the train_model.ipynb notebook

# Result
result includes the resume's predicted job category, providing quick and accurate predictions for job classification based on the uploaded resume.




