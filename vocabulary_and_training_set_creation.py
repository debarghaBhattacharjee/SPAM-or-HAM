#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *

import time
import pandas as pd
import numpy as np


# In[4]:


start_time = time.time()


# In[5]:


df = pd.read_csv(
    filepath_or_buffer="resources/collection_frequency.csv", 
    sep=",", header=0
)
vocabulary = df["WORD"]
print(df)


# In[6]:


word_to_id = {}
id_to_word = {}

for idx, word in enumerate(sorted(vocabulary)):
    word_to_id[word] = idx
    id_to_word[idx] = word
vocabulary_size = len(word_to_id)

print(f"Total unique words in the vocabulary: {vocabulary_size}")


# In[7]:


DATA_DIR = "enron1/training"
CATEGORIES = ["ham", "spam"]


# In[8]:


total_hams = 0
total_spams = 0
total_emails = 0

for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    if category == "ham":
        total_hams = len([email for email in os.listdir(path)])
    elif category == "spam":
        total_spams = len([email for email in os.listdir(path)])
        
total_emails = total_hams + total_spams

print("==============================================")
print("STATISTICS (TRAINING SET):")
print("----------------------------------------------")
print(f"Total emails: {total_emails}")
print(f"Total non-spam emails: {total_hams}")
print(f"Total spam emails: {total_spams}")
print("==============================================")


# In[9]:


# Plot training set stats related to 
# distribution of spam and non-spam emails.
title =     "Distribution of spam and ham (non-spam) emails in training set"

class_labels_train = ["ham", "spam"]
class_counts_train = [total_hams, total_spams]

file_dir = "resources"
file_name = "spam_ham_distribution_training_set.pdf"

plot_spam_ham_distribution(
    title = title,
    class_labels=class_labels_train,
    class_counts=class_counts_train,
    file_dir=file_dir,
    file_name=file_name
)


# In[10]:


term_frequency_matrix = np.zeros((vocabulary_size, total_emails))
email_labels = np.zeros(total_emails)


# In[11]:


lemmatizer = WordNetLemmatizer()
current_email_nb = 0 # Keeps track of current email number being read.
emails_read = 0 # Keeps track of total emails read till now.
hams_read = 0 # Keeps track of total non-spam emails read till now.
spams_read = 0 # Kepps track og toal spam mails read till now.

for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    for email in os.listdir(path):
        file = open(os.path.join(path, email), "r", encoding="utf-8", errors="ignore")
        text = file.read()       
        cleaned_text = text_cleanup(text)
        file.close()
        
        for word in cleaned_text:
            word = lemmatizer.lemmatize(word)
            if word in word_to_id:
                term_frequency_matrix[word_to_id[word], current_email_nb] += 1
                
        if category == "ham":
            email_labels[current_email_nb] = -1
        elif category == "spam":
            email_labels[current_email_nb] = 1
            
        current_email_nb += 1
        emails_read += 1
        if emails_read % 100 == 0:
            print(f"{emails_read} emails read.")
            
    if category == "ham":
        hams_read = emails_read
    elif category == "spam":
        spams_read = emails_read - hams_read
        
print("==============================================")
print("STATISTICS:")
print("----------------------------------------------")
print(f"Total emails read: {emails_read}")
print(f"Total non-spam emails read: {hams_read}")
print(f"Total spam emails read: {spams_read}")
print(f"Total unique words: {vocabulary_size}")
print("==============================================")


# In[12]:


# Transpose and take log of every element.
term_frequency_matrix = term_frequency_matrix.T

log_term_frequency_matrix =     np.log(term_frequency_matrix + 1)
print(log_term_frequency_matrix.shape)


# In[13]:


# Create the inverse document frequency vector from the training set.
inverse_document_frequency_vector = np.zeros(vocabulary_size)
for i in range(vocabulary_size):
    inverse_document_frequency_vector[i] =         total_emails / np.count_nonzero(term_frequency_matrix[:, i])

log_inverse_document_frequency_vector =     np.log(inverse_document_frequency_vector)


# In[14]:


# Save the vocabulary.
if not os.path.isdir("resources"):
    os.makedirs("resources")
    
f = open("resources/vocabulary.csv", "w+")

# Write top headers.
f.write("WORD,ID,INVERSE DOCUMENT FREQUENCY")
f.write("\n")

for word in word_to_id.keys():
    f.write(f"{word},{word_to_id[word]},{log_inverse_document_frequency_vector[word_to_id[word]]}")
    f.write("\n")

f.close()


# In[15]:


# Create tf-idf matrix.
tf_idf_matrix = np.zeros(log_term_frequency_matrix.shape)
for i in range(total_emails):
    tf_idf_matrix[i, :] =         log_term_frequency_matrix[i, :] *         log_inverse_document_frequency_vector


# In[16]:


# Save training set as a csv file.
if not os.path.isdir("resources"):
    os.makedirs("resources")

f = open("resources/training_set.csv", "w+")

# Write top headers.
f.write("EMAIL,")
for word in word_to_id.keys():
    f.write(f"{word},")
f.write("LABEL")
f.write("\n")

for i in range(total_emails):
    f.write(f"EMAIL {i},")
    for j in range(vocabulary_size):
        f.write(f"{tf_idf_matrix[i, j]},")
    f.write(f"{email_labels[i]}")
    f.write("\n") 

f.close()


# In[17]:


print(
    "Total time (in seconds) taken to create vocabulary and training set: " + \
    f"{round(time.time() - start_time, 2)}"
)

