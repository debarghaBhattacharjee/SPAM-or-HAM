#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import string
import pickle
import matplotlib.pyplot as plt

import nltk
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


# In[2]:


def text_cleanup(text):
    """
    Method to clean the input text by-
    1. Removing punctuations.
    2. Removing numbers.
    3. Removing stopwords.
    4. Lowercasing the words.
    5. Removing words having length <= 2. 
    """
    
    # Removes the punctuation from the original text.
    punctuation_removed_text = [
        word
        for word in text 
        if word not in string.punctuation
    ]
    punctuation_removed_text = "".join(punctuation_removed_text)
    
    # Removes numbers from the punctuation removed text.
    numbers_removed_text = [
        word
        for word in punctuation_removed_text.split()
        if not word.isdigit()
    ]
    numbers_removed_text = " ".join(numbers_removed_text)
    
    # Removes the stopwords from the numbers removed text.
    stopwords_removed_text = [
        word
        for word in numbers_removed_text.split()
        if word.lower() not in stopwords.words("english")
    ]
    stopwords_removed_text = " ".join(stopwords_removed_text)
    
    # Converts all uppercase alphabets to their lowercase counterparts.
    lowercased_text = [
        word.lower()
        for word in stopwords_removed_text.split()
    ]
    lowercased_text = " ".join(lowercased_text)
    
    # Create cleaned text by retaining words having length greater
    # than 2.
    cleaned_text = [
        word
        for word in lowercased_text.split()
        if len(word) > 2
    ]
    
    return cleaned_text


# In[3]:


def plot_spam_ham_distribution(title, class_labels, class_counts, file_dir=None, file_name=None):
    """
    Method to plot the stats related to distribution of spam and
    non-spam email distribution.
    """
    # Create directory if it does not exist.
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    
    fig = plt.figure(figsize=(10, 7))
    plt.title(title)
    plt.pie(class_counts, labels=class_labels, autopct='%1.2f%%')
    plt.savefig(f"{file_dir}/{file_name}")
    plt.show()
    plt.close()


# In[4]:


def save_data(data, file_name, directory):
    """
    Metho to save data as a pickled string.
    """
    # Create directory if it does not exist.
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # Save data as pickled object.
    pickle_out = open(
        f"{directory}/{file_name}", 
        "wb"
    )
    pickle.dump(data, pickle_out)
    pickle_out.close()
    
    msg = f"Model successfully saved: {directory}/{file_name}"
    print(msg)


# In[5]:


def load_data(data_path):
    """
    Method to load data saved a s pickled string.
    """
    # Create directory if it does not exist.
    if not os.path.exists(data_path):
        print("Data doesn't exist.")
        return None
    
    # Retrieve the file.
    pickle_in = open(data_path,"rb")
    retrieved_data = pickle.load(pickle_in)
    pickle_in.close()
    return retrieved_data


# In[ ]:




