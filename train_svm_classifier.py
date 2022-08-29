#!/usr/bin/env python
# coding: utf-8

# In[8]:


from utils import *

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import metrics, svm


# In[9]:


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


# In[10]:


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


# In[11]:


np.random.seed(1997)


# In[12]:


df_vocab = pd.read_csv(
    filepath_or_buffer="resources/vocabulary.csv", 
    sep=",", header=0
)
vocabulary = df_vocab["WORD"]
idf_vector = df_vocab["INVERSE DOCUMENT FREQUENCY"].to_numpy()
print(df_vocab)


# In[13]:


word_to_id = {}
id_to_word = {}

for idx, word in enumerate(sorted(vocabulary)):
    word_to_id[word] = idx
    id_to_word[idx] = word
vocabulary_size = len(word_to_id)

print(f"Total unique words in the vocabulary: {vocabulary_size}")


# In[14]:


df_train = pd.read_csv(
    filepath_or_buffer="resources/training_set.csv", 
    sep=",", header=0
)
print(df_train)


# In[15]:


df_train_np = df_train.to_numpy()[0:, 1:]
np.random.shuffle(df_train_np)
print(df_train_np.shape)


# In[16]:


x_train = df_train_np[:, :-1]
n_train, d = x_train.shape
print("==============================================")
print("INPUT DATA (TRAINING): ")
print("----------------------------------------------")
print(x_train)
print("----------------------------------------------")
print(f"SHAPE: {x_train.shape}")
print(f"# data points: {n_train}")
print(f"# features: {d}")
print("==============================================")
print()


# In[17]:


y_train = df_train_np[:, -1]
y_train = y_train.astype('int')
print("==============================================")
print("TARGET LABELS (TRAINING): ")
print("----------------------------------------------")
print(y_train)
print("----------------------------------------------")
print(f"SHAPE: {y_train.shape}")
print(f"# data points: {len(y_train)}")
print("==============================================")
print()


# In[18]:


df_cv = pd.read_csv(
    filepath_or_buffer="resources/cross_validation_set.csv", 
    sep=",", header=0
)
print(df_cv)


# In[19]:


df_cv_np = df_cv.to_numpy()[0:, 1:]
print(df_cv_np.shape)


# In[20]:


x_cv = df_cv_np[:, :-1]
n_cv, d = x_cv.shape
print("==============================================")
print("INPUT DATA (CROSS-VALIDATION): ")
print("----------------------------------------------")
print(x_cv)
print("----------------------------------------------")
print(f"SHAPE: {x_cv.shape}")
print(f"# data points: {n_cv}")
print(f"# features: {d}")
print("==============================================")
print()


# In[21]:


y_cv = df_cv_np[:, -1]
y_cv = y_cv.astype('int')
print("==============================================")
print("TARGET LABELS (CROSS-VALIDATION): ")
print("----------------------------------------------")
print(y_cv)
print("----------------------------------------------")
print(f"SHAPE: {y_cv.shape}")
print(f"# data points: {len(y_cv)}")
print("==============================================")
print()


# In[22]:


# Different values of regularization parameter c.
reg_param_values = np.arange(0, 100, 10)
reg_param_values[0] = 1
nb_reg_param_values = len(reg_param_values)
print(
    f"Testing linear SVM  model for following " + \
    f"{nb_reg_param_values} values of regularization paramater(c): "
)
print(reg_param_values)

# Array to store (training and cross-validation) accuracy scores.
accuracy_scores_train = np.zeros(nb_reg_param_values)
accuracy_scores_cv = np.zeros(nb_reg_param_values)

# Array to store cross-validation precision scores.
precision_scores_cv = np.zeros(nb_reg_param_values)

# Array to store cross-validation recall scores.
recall_scores_cv = np.zeros(nb_reg_param_values)

# Array to store cross-validation f1-scores.
f1_scores_cv = np.zeros(nb_reg_param_values)


# In[ ]:


# Train using linear SVM classifier.
svc_best = None # Saves classifier which gives best cv f1-score.
idx_best = 0 # Saves index of corresponding c from list.
max_f1_score_cv_seen = 0.0 # Stores max. cv f1-score seen.

for i in tqdm(range(nb_reg_param_values), unit=" it"):
    c = reg_param_values[i]
    svc = svm.SVC(C=c)
    svc.fit(x_train, y_train)
    
    accuracy_scores_train[i] = svc.score(x_train, y_train)
    accuracy_scores_cv[i] = svc.score(x_cv, y_cv)
    
    precision_scores_cv[i] = metrics.precision_score(y_cv, svc.predict(x_cv))
    recall_scores_cv[i] = metrics.recall_score(y_cv, svc.predict(x_cv))
    f1_scores_cv[i] = metrics.f1_score(y_cv, svc.predict(x_cv))
    
    if f1_scores_cv[i] >= max_f1_score_cv_seen:
        # Update best classifier.
        svc_best = svc
        idx_best = i
        max_f1_score_cv_seen = f1_scores_cv[i]


# In[ ]:


results = np.matrix(
    np.c_[
        reg_param_values, 
        accuracy_scores_train, accuracy_scores_cv, 
        precision_scores_cv, recall_scores_cv,
        f1_scores_cv
    ]
)

results_df = pd.DataFrame(
    data = results, 
    columns = [
        'Reg. Parameter (c)', 
        'Train Accuracy', 'Cross-Validation Accuracy', 
        'Cross-Validation Precision', 'Cross-Validation Recall',
        'Cross-Validation F1-Score'
    ]
)


# In[ ]:


print("==============================================")
print("HYPERPARAMETER TUNING TEST RESULT: ")
print("----------------------------------------------")
print(results_df.head(n=10))
print("----------------------------------------------")
print("SELECTED MODEL STATS: ")
print("----------------------------------------------")
print("Classifier type: Linear SVM Classifier")
print(f"Regularization parameter value (c): {reg_param_values[idx_best]}")
print("----------------------------------------------")
print(f"Train accuracy: {accuracy_scores_train[i]}")
print(f"Cross-valididation accuracy: {accuracy_scores_cv[idx_best]}")
print(f"Cross-valididation precision: {precision_scores_cv[idx_best]}")
print(f"Cross-valididation recall: {recall_scores_cv[idx_best]}")
print(f"Cross-valididation f1-Score: {f1_scores_cv[idx_best]}")
print("==============================================")


# In[ ]:


svm_model_data = {
    "model" : svc_best,
    "classifier_type" : "Linear SVM Classifier",
    "word_to_id" : word_to_id,
    "id_to_word" : id_to_word,
    "idf_vector" : idf_vector,
}

data = svm_model_data
model_name = "svm_model_data_collection_frequency_threshold-5"
model_dir = "models"
save_data(
    data=svm_model_data,
    file_name=model_name,
    directory=model_dir
) 


# In[ ]:




