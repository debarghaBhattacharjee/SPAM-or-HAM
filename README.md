# SPAM-or-HAM
A simple spam filter leveraging SVM and Naive Bayes algorithm to detect spam and non-spam emails from the enron1 dataset.

## Instructions

Please follow the steps given below-

1. Copy 'enron1' (which contains training and test emails), 'test' (directory which contains all test mails) and all the python scripts to the same directory.
2. **TRAINING PHASE-** Run the scripts in the following order-
	(a) Run *collection_frequency.py*
	(b) Run *vocabulary_and_training_set_creation.py*
	(c) Run *cross_validation_set_creation.py*
	(d) Run *train_svm_classifier.py*
3. **TESTING PHASE-** Run *predict_emails.py* to predict emails in test set.

Link for downloading dataset- https://drive.google.com/file/d/1_G2175I1DAawnfGlIMdMAcgy3Sy5iwL2/view?usp=sharing

**Created by-** <br>
	*Debargha Bhattacharjee* <br>
	*CS19S028, MS Scholar* <br>
	*Department of Computer Science and Engineering* <br>
	*IIT Madras* <br>
	*CS5691 Pattern Recognition and Machine Learning Course Project* <br>
