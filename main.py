from sklearn import metrics, svm
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import subprocess
from utils import *


def main(): 
    saved_svm_model_data = load_data(
        "models/svm_model_data_collection_frequency_threshold-5")
    svc = saved_svm_model_data["model"]
    word_to_id = saved_svm_model_data["word_to_id"]
    id_to_word = saved_svm_model_data["id_to_word"]
    idf_vector = saved_svm_model_data["idf_vector"]
    vocabulary_size = len(word_to_id)
    # Read the emails in test set.
    DATA_DIR = "test"
    # Count the total number of emails in test set.
    email_names = [email for email in os.listdir(DATA_DIR)]
    total_emails = len(email_names)
    term_frequency_matrix = np.zeros((total_emails, vocabulary_size))
    predicted_scores = np.zeros(total_emails)
    lemmatizer = WordNetLemmatizer()
    current_email_nb = 0  # Keeps track of current email number being read.
    emails_read = 0  # Keeps track of total emails read till now.
    # Read and clean emails.
    # Then, create bow term frequency matrix.  
    for email in os.listdir(DATA_DIR):
        file = open(os.path.join(DATA_DIR, email), "r",
                    encoding="utf-8", errors="ignore")
        text = file.read()
        cleaned_text = text_cleanup(text)
        file.close()

        for word in cleaned_text:
            word = lemmatizer.lemmatize(word)
            if word in word_to_id:
                term_frequency_matrix[current_email_nb, word_to_id[word]] += 1

        current_email_nb += 1
        emails_read += 1
       
    log_term_frequency_matrix = np.log(term_frequency_matrix + 1)

    # Create tf-idf matrix.
    tf_idf_matrix = np.zeros(log_term_frequency_matrix.shape)
    for i in range(total_emails):
        tf_idf_matrix[i, :] = log_term_frequency_matrix[i, :] * idf_vector

    # Predict emails as either spam or ham.
    predicted_scores = svc.predict(tf_idf_matrix)
    predicted_labels = [
        'spam' if predicted_score == 1 else 'ham'
        for predicted_score in predicted_scores
    ]




    results = np.matrix(
        np.c_[email_names, predicted_scores, predicted_labels]
    )

    results_df = pd.DataFrame(
        data=results,
        columns=[
            'Email',
            'Predicted Score',
            'Predicted Label'
        ]
    )

    #print("Displaying predicted labels of first 10 emails-")

    print([i if i == 1 else 0 for i in predicted_scores])



if __name__ == "__main__":
    main()

