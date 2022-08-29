#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sklearn import metrics, svm


# In[2]:


saved_svm_model_data = load_data("models/svm_model_data_collection_frequency_threshold-5")


# In[3]:


svc        = saved_svm_model_data["model"]
word_to_id = saved_svm_model_data["word_to_id"]
id_to_word = saved_svm_model_data["id_to_word"]
idf_vector = saved_svm_model_data["idf_vector"]
vocabulary_size = len(word_to_id)


# In[4]:


# Read the emails in test set.
DATA_DIR = "test"

# Count the total number of emails in test set.
email_names = [email for email in os.listdir(DATA_DIR)]
total_emails = len(email_names)
print(f"Total no. of emails in test set: {total_emails}")

term_frequency_matrix = np.zeros((total_emails, vocabulary_size))
predicted_scores = np.zeros(total_emails)


# In[5]:


lemmatizer = WordNetLemmatizer()
current_email_nb = 0 # Keeps track of current email number being read.
emails_read = 0 # Keeps track of total emails read till now.

# Read and clean emails.
# Then, create bow term frequency matrix.
print("Reading emails from test set.")
for email in os.listdir(DATA_DIR):
    file = open(os.path.join(DATA_DIR, email), "r", encoding="utf-8", errors="ignore")
    text = file.read()       
    cleaned_text = text_cleanup(text)
    file.close()
    
    for word in cleaned_text:
        word = lemmatizer.lemmatize(word)
        if word in word_to_id:
            term_frequency_matrix[current_email_nb, word_to_id[word]] += 1
                
    current_email_nb += 1
    emails_read += 1
    if emails_read % 10 == 0:
        print(f"{emails_read} emails read.")


# In[6]:


# Take log of every element.
log_term_frequency_matrix =     np.log(term_frequency_matrix + 1)

# Create tf-idf matrix.
tf_idf_matrix = np.zeros(log_term_frequency_matrix.shape)
for i in range(total_emails):
    tf_idf_matrix[i, :] = log_term_frequency_matrix[i, :] * idf_vector


# In[7]:


# Predict emails as either spam or ham.
predicted_scores = svc.predict(tf_idf_matrix)
predicted_labels = [
    'spam' if predicted_score == 1 else 'ham'
    for predicted_score in predicted_scores
]


# In[11]:


results = np.matrix(
    np.c_[email_names, predicted_scores, predicted_labels]
)

results_df = pd.DataFrame(
    data = results, 
    columns = [
        'Email', 
        'Predicted Score', 
        'Predicted Label'
    ]
)

print("Displaying predicted labels of first 10 emails-")
print(results_df.head(n=10))


# In[ ]:




