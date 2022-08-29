#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *

import time
import operator


# In[2]:


start_time = time.time()


# In[3]:


DATA_DIR = "enron1/training"
CATEGORIES = ["ham", "spam"]


# In[4]:


lemmatizer =  WordNetLemmatizer()
emails_read = 0
hams_read = 0
spams_read = 0
count = {}


# In[5]:


# Iterate through the emails in the training set 
# to extract unique words.
for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    for email in os.listdir(path):
        file = open(os.path.join(path, email), "r", encoding="utf-8", errors="ignore")
        text = file.read()       
        cleaned_text = text_cleanup(text)
        file.close()
        
        for word in cleaned_text:
            word = lemmatizer.lemmatize(word)
            if word in count:
                count[word] += 1
            else:
                count[word] = 1
                
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
print("==============================================")


# In[6]:


# Sort the count dictionary according to the frequency of 
# occurence of different words in the dictionary.
sorted_count = sorted(
    count.items(), 
    key=operator.itemgetter(1),
    reverse=True
)

sorted_count = dict(sorted_count)


# In[7]:


# Create a csv containing collection frequency  of words 
# having collection frequency >= 20.
if not os.path.isdir("resources"):
    os.makedirs("resources")

f = open("resources/collection_frequency.csv", "w+")
f.write("WORD,COLLECTION FREQUENCY")
f.write("\n")
for word, frequency in sorted_count.items():
    if frequency >= 5:
#         f.write(f"{str(word)},{str(frequency)}")
        f.write(f"{word},{frequency}")
        f.write("\n")
f.close()


# In[8]:


print(
    "Total time (in seconds) to create collection frequency: " + \
    f"{round(time.time() - start_time, 2)}"
)


# In[ ]:




