# vaccine-tweets-ro-research

# Observation: Please note that due to the data protection regulations, the dataset containing the collected and processed tweets is available only in a Tfidf vectorized format (the one required for the machine learning algorithm development), in order to avoid any direct or indirect identification of the Twitter accounts from which the content was created.

# Filename description

# Data processing_final.py = the Python code for the preprocessing of the 1400 collected tweets, wordcloud representation, correlation analysis, the development and validation of the machine learning scikit-learn predictive models.

# Data processing_final_RCNN.py = the Python code for the preprocessing of the 1400 collected tweets, wordcloud representation, correlation analysis, the development and validation of the predictive model based on recurrent convolutional neural networks (RCNN) - Tensorflow implementation.

# Data processing_final_BERT.py = the Python code for the preprocessing of the 1400 collected tweets, wordcloud representation, correlation analysis, the development and validation of the predictive model based on BERT (https://huggingface.co/dumitrescustefan/bert-base-romanian-uncased-v1) - Tensorflow implementation.

# Internal_data_vectors.csv = the TFidf vectorized data corresponding to the tweets from the internal dataset (1300 tweets)
# Internal_data_labels.csv = the labels resulted from the manual annotation of the tweets from the internal dataset (1300 tweets)
# Internal_data_ID.csv = the tweet IDs for the internal dataset (1300 tweets)

# External_data_vectors.csv = the TFidf vectorized data corresponding to the tweets from the external dataset (100 tweets)
# External_data_labels.csv = the labels resulted from the manual annotation of the tweets from the external dataset (100 tweets)
# External_data_ID.csv = the tweet IDs for the external dataset (100 tweets)

# Labels explanation (for the two csv files containing the tweet labels): 0 = true content; 1 = neutral content; 2 = fake content

# Model_SVC.sav = the predictive Support Vector Machines model based on the 1300 annotated tweets from the internal dataset; the algorithm was evaluated based on its ability to estimate the probability that a specific tweet is true, neutral and fake, as well as on its ability to classify a specific tweet in true, neutral or fake.
