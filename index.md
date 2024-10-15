---
layout: default
---

# Projects

## Project 1: Automatic Ticket Classification

**Problem Statement:**  
The goal of this project is to automate the classification of customer complaints for a financial company, using unstructured text data from customer support tickets. The company offers various products and services, such as credit cards, banking, and mortgages/loans. The objective is to use non-negative matrix factorization (NMF), a topic modeling technique, to detect patterns and recurring words in the complaints, grouping them into five categories: Credit card/Prepaid card, Bank account services, Theft/Dispute reporting, Mortgages/loans, and Others. Once categorized, this data will be used to train supervised models (e.g., logistic regression, decision trees, or random forests) to classify future complaints into their relevant departments, improving response times and customer satisfaction.

**Technologies / Libraries Used:**  
Python Pandas, NumPy, Matplotlib, Seaborn, Plotly, sklearn, XGBoost

**Models:**  
Logistic Regression, Decision Tree, Random Forest, Multinomial Naive Bayes, XGBoost Classifier

**Code:**  
[GitHub - Automatic Ticket Classification](https://github.com/avinashb7/AutomaticTicketClassification/blob/main/Automatic_Ticket_Classification_Assignment.ipynb)

---

## Project 2: Melanoma Cancer Detection

**Problem Statement:**  
The goal of this project is to develop a custom convolutional neural network (CNN) model using TensorFlow to accurately detect melanoma from skin images. Given that melanoma accounts for 75% of skin cancer deaths, an effective solution can significantly aid dermatologists in diagnosing this deadly disease. The model will analyze a dataset of 2,357 images of various skin conditions, including melanoma, to classify them accurately.

**Technologies / Libraries Used:**  
Python Pandas, NumPy, TensorFlow, Keras

**Models:**  
CNN - Sequential model, Used Augmentor

**Code:**  
[GitHub - Melanoma Cancer Detection](https://github.com/avinashb7/MelonomaCancerAssignment/blob/main/melonoma_cancer.ipynb)

**Project:**  
[GitHub Repository](https://github.com/avinashb7/MelonomaCancerAssignment/tree/main)

---

## Project 3: Gesture Recognition

**Problem Statement:**  
The goal of the project is to develop a gesture recognition feature for smart TVs that allows users to control the TV without a remote by recognizing five specific hand gestures using the built-in webcam. Each gesture corresponds to an action such as increasing or decreasing volume, jumping forward or backward in a video, or pausing playback.

**Technologies / Libraries Used:**  
Python Pandas, NumPy, Matplotlib, Keras, Imageio

**Models:**  
Conv3D with varying epochs and batch size, CNN-LSTM Model, Transfer Learning

**Code:**  
[GitHub - Gesture Recognition](https://github.com/avinashb7/Gesture-Recognition/blob/main/Gesture%20Recognition%20Project%20Upgrad.ipynb)

**Project:**  
[GitHub Repository](https://github.com/avinashb7/Gesture-Recognition)

---

## Project 4: HelpMate AI (Generative AI)

**Problem Statement:**  
This project aims to develop a semantic search system utilizing the RAG (Retrieval-Augmented Generation) pipeline for efficient document retrieval from PDF sources. The project extracts relevant information from PDF documents and organizes it into a structured format, creating vector representations with the SentenceTransformerEmbedding's all-MiniLM-L6-v2 model. A caching layer is integrated to store and retrieve previous queries and their results. The RAG pipeline consists of three main layers: an Embedding Layer, a Search and Rank Layer for semantic similarity searches, and a Generation Layer that generates answers based on user queries.

**Technologies / Libraries Used:**  
pdfplumber, chromaDB, Pandas, OpenAI, pathlib

**Steps:**  
PDF Chunking, Generating and Storing embeddings using OpenAI and ChromaDB, Semantic Search with Cache, Re-Ranking with Cross Encoder, RAG

**Code:**  
[GitHub - HelpMate AI](https://github.com/avinashb7/HelpMate_GenAI/blob/main/Generative_Search_with_SentenceTransformers_and_Chroma.ipynb)

**Project:**  
[GitHub Repository](https://github.com/avinashb7/HelpMate_GenAI/tree/main)

---

## Project 5: Telecom Churn Case Study

**Problem Statement:**  
The goal of this project is to develop a machine learning model to predict customer churn for a leading telecom company, using customer-level data over four months. The project focuses on understanding customer behavior in different lifecycle phases—'good,' 'action,' and 'churn'—to predict churn and enable the company to take corrective actions in the 'action' phase to retain high-value customers.

**Technologies / Libraries Used:**  
Python Pandas, NumPy, Missingno, XGBoost, sklearn, lightgbm, seaborn, matplotlib

**Models:**  
Logistic Regression, Decision Tree, Random Forest Classifier, Gradient Boosting, SVC (Support Vector Classifier), LightGBM

**Code:**  
[GitHub - Telecom Churn Case Study](https://github.com/avinashb7/TelecomChurnCaseStudy/blob/main/Telecom_churn_case_study.ipynb)

**Project:**  
[GitHub Repository](https://github.com/avinashb7/TelecomChurnCaseStudy)

---

## Project 6: Lending Club Case Study

**Problem Statement:**  
This project aims to identify patterns that indicate if a person is likely to default on a loan. Such insights may be used to take actions like denying the loan, reducing the loan amount, or lending to risky applicants at a higher interest rate.

**Technologies / Libraries Used:**  
Python Pandas, Seaborn, Matplotlib

**Code:**  
[GitHub - Lending Club Case Study](https://github.com/avinashb7/Lending_Club_CaseStudy/blob/main/Lending_club_case_study.ipynb)

**Project:**  
[GitHub Repository](https://github.com/avinashb7/Lending_Club_CaseStudy/tree/main)

