## Import modules

from textblob.classifiers import DecisionTreeClassifier as NBC
from textblob import TextBlob
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import nltk
import fasttext
import textract
import matplotlib.pyplot as plt
import matplotlib
import joblib
import unidecode
import spacy
from sklearn.preprocessing import LabelBinarizer
import statsmodels
from statsmodels.stats import inter_rater as irr
import krippendorff as kd
import scattertext as st
from scipy.stats import mode
import scipy.stats
import scipy.sparse as sp
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import unidecode
import re

import string
from spacy.lang.ro import Romanian
from spacy.lang.ro.stop_words import STOP_WORDS
punctuations = string.punctuation

import ro_core_news_sm
nlp = ro_core_news_sm.load()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Embedding, Bidirectional, MaxPooling1D,BatchNormalization,Dropout


## Import the collected tweets - internal data (full text - column no.3, manual annotation - for the 3 annotators - columns 12, 13 and 14).

Predictor_total = np.array(pd.read_excel("Echipa_1+2+3_21062022.xlsx",usecols = (3,12,13,14),header = None,dtype = str,skiprows = 1))

## Import the engagement metrics for the collected tweets - internal data (columns no. 6, 7 and 8: number of replies, number of retweets, number of likes).

Retweet_total = np.array(pd.read_excel("Echipa_1+2+3_21062022.xlsx",usecols = (6,7,8),header = None,dtype = float,skiprows = 1))

## Convert the manual annotation of the tweets from string to float datatype and eliminate the tweets written in other languages (not Romanian).

Predictor_total[Predictor_total[:,1]=='True',1] = '0'
Predictor_total[Predictor_total[:,1]=='FAKE',1] = '2'
Predictor_total[Predictor_total[:,1]=='NONE',1] = '1'

Predictor_total[Predictor_total[:,2]=='True',2] = '0'
Predictor_total[Predictor_total[:,2]=='FAKE',2] = '2'
Predictor_total[Predictor_total[:,2]=='NONE',2] = '1'

Predictor_total[Predictor_total[:,3]=='True',3] = '0'
Predictor_total[Predictor_total[:,3]=='FAKE',3] = '2'
Predictor_total[Predictor_total[:,3]=='NONE',3] = '1'

Hospz_test = Predictor_total[:,1].astype(float)
Predictor_total = Predictor_total[np.isnan(Hospz_test)==False,:]
Retweet = Retweet_total[np.isnan(Hospz_test)==False,:]

## Data preprocessing: delete the words starting with @, since they correspond to a specific username, delete the URLs,
#convert all characters to lowercase, eliminate stopwords and punctuations.

for i in range (0,len(Predictor_total[:,0])):
    a = Predictor_total[i,0]
    Predictor_total[i,0] = " ".join(filter(lambda x:x[0]!='@', a.split()))

for i in range (0,len(Predictor_total[:,0])):
    a = Predictor_total[i,0]
    Predictor_total[i,0] = re.sub(r'http\S+', '',a)


text = Predictor_total[:,0]
for i in range (0,len(text)):
    text[i] = str(nlp(str(text[i])))
    a = nlp(str(text[i]))
    b = [word.lower_ for word in a]
    c = [ word for word in b if word not in STOP_WORDS and word not in punctuations ]
    Predictor_total[i,0] = str(c)
    Predictor_total[i,0] = unidecode.unidecode(Predictor_total[i,0])

## Create the final class for each of the manually annotated tweets, based on the majority vote rule.

Delete = np.zeros((len(Predictor_total[:,0]),))
Majority = np.empty((len(Predictor_total[:,0]),),dtype=float)
for i in range (0,len(Predictor_total[:,0])):
    unique,counts = np.unique(Predictor_total[i,1:4],return_counts = True)
    if len(unique)<3:
        Majority[i] = float(unique[counts==np.max(counts)])
    elif len(unique)==3 and Retweet[i,1]!=0:
        Majority[i]=1
    else:
        Majority[i]=4


Predictor_total = Predictor_total[Majority!=4,:]
Retweet = Retweet[Majority!=4,:]
Majority = Majority[Majority!=4]

print(Majority)
Hospz = Majority.reshape(-1,1)
Hospz_str = Majority.astype(str).reshape(-1,1)

for k in range (0,len(Hospz_str)):
    Hospz_str[k] = "{:.0f}".format(Majority[k])

Predictor = Predictor_total[:,0].reshape(-1,1)


## Wordcloud representation of word frequency for true, neutral and fake vaccine tweets. The most relevant 30 words and word combinations (as set by the max_words parameter)
# are printed and graphically represented for each of the three classes.

nlp = ro_core_news_sm.load()

Predictor_1 = Predictor.astype(str)

from wordcloud import WordCloud

for k in range (0,3):
    wc = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Set2', collocations=True,collocation_threshold = 0,normalize_plurals = False,
                   max_words = 30).generate(str(Predictor[Majority==k]))
    print(wc.words_)
    plt.axis("off")
    plt.imshow(wc, interpolation="bilinear")
    plt.show()

### End of script for the wordcloud representation


### Reimporting of the tweets for correlation analysis and machine learning training

## Import the collected tweets - internal data (full text, manual annotation).

Predictor_total = np.array(pd.read_excel("Echipa_1+2+3_21062022.xlsx",usecols = (3,12,13,14),header = None,dtype = str,skiprows = 1))

## Import the engagement metrics for the collected tweets - internal data (number of replies, number of retweets, number of likes)

Retweet_total = np.array(pd.read_excel("Echipa_1+2+3_21062022.xlsx",usecols = (6,7,8),header = None,dtype = float,skiprows = 1))

## Import the date of the collected tweets - internal data.

Date_total = np.array(pd.read_excel("Echipa_1+2+3_21062022.xlsx",usecols = (1,),header = None,dtype = 'datetime64[D]',skiprows = 1))


## Convert the manual annotation of the tweets from string to float datatype and eliminate the tweets written in other languages (not Romanian).

Predictor_total[Predictor_total[:,1]=='True',1] = '0'
Predictor_total[Predictor_total[:,1]=='FAKE',1] = '2'
Predictor_total[Predictor_total[:,1]=='NONE',1] = '1'

Predictor_total[Predictor_total[:,2]=='True',2] = '0'
Predictor_total[Predictor_total[:,2]=='FAKE',2] = '2'
Predictor_total[Predictor_total[:,2]=='NONE',2] = '1'

Predictor_total[Predictor_total[:,3]=='True',3] = '0'
Predictor_total[Predictor_total[:,3]=='FAKE',3] = '2'
Predictor_total[Predictor_total[:,3]=='NONE',3] = '1'

Hospz_test = Predictor_total[:,1].astype(float)
Predictor_total = Predictor_total[np.isnan(Hospz_test)==False,:]
Retweet = Retweet_total[np.isnan(Hospz_test)==False,:]
Date = Date_total[np.isnan(Hospz_test)==False]

print(Predictor_total.shape)

## Data preprocessing (see the details listed above).

for i in range (0,len(Predictor_total[:,0])):
    a = Predictor_total[i,0]
    Predictor_total[i,0] = " ".join(filter(lambda x:x[0]!='@', a.split()))

for i in range (0,len(Predictor_total[:,0])):
    a = Predictor_total[i,0]
    Predictor_total[i,0] = re.sub(r'http\S+', '',a)

Predictor_total_1 = Predictor_total
Retweet_1 = Retweet
Date_1 = Date

Predictor_total_2 = Predictor_total
Retweet_2 = Retweet
Date_2 = Date

News = Predictor_total_1[:,0]
No_words_1 = np.zeros((len(News),))

for i in range (0,len(News)):
    Word_list = str(News[i]).split()
    No_words_1[i] = len(Word_list)

text = Predictor_total_1[:,0]
Vectors_1 = np.zeros((len(text),96))
for i in range (0,len(text)):
    text[i] = str(nlp(str(text[i])))
    a = nlp(str(text[i]))
    b = [word.lower_ for word in a]
    c = [ word for word in b if word not in STOP_WORDS and word not in punctuations ]
    Vectors_1[i,:] = nlp(str(text[i])).vector
    Predictor_total_1[i,0] = str(c)
    Predictor_total_1[i,0] = unidecode.unidecode(Predictor_total_1[i,0])

No_words_2 = No_words_1
Vectors_2 = Vectors_1

## Create the final class for each of the manually annotated tweets, based on the majority vote rule.

Delete = np.zeros((len(Predictor_total[:,0]),))
Majority = np.empty((len(Predictor_total[:,0]),),dtype=float)
for i in range (0,len(Predictor_total[:,0])):
    unique,counts = np.unique(Predictor_total[i,1:4],return_counts = True)
    if len(unique)<3:
        Majority[i] = float(unique[counts==np.max(counts)])
    elif len(unique)==3 and Retweet[i,1]!=0:
        Majority[i]=1
    else:
        Majority[i]=4

Predictor_total = Predictor_total[Majority!=4,:]
Retweet = Retweet[Majority!=4,:]
Date = Date[Majority!=4]
Majority = Majority[Majority!=4]


text = Predictor_total[:,0]
Vectors = np.zeros((len(text),96))
for i in range (0,len(text)):
    text[i] = str(nlp(str(text[i])))
    a = nlp(str(text[i]))
    b = [word.lower_ for word in a]
    c = [ word for word in b if word not in STOP_WORDS and word not in punctuations ]
    Vectors[i,:] = nlp(str(text[i])).vector
    Predictor_total[i,0] = str(c)
    Predictor_total[i,0] = unidecode.unidecode(Predictor_total[i,0])
    
print(Majority.shape)
Hospz = Majority.reshape(-1,1)
Hospz_1d = Majority
Hospz_str = Majority.astype(str).reshape(-1,1)

Predictor = Predictor_total[:,0].reshape(-1,1)

## Correlations between tweet engagement metrics (including number of words) and classification (true, neutral, fake).

News = Predictor
No_words = np.zeros((len(News),))

for i in range (0,len(News)):
    Word_list = str(News[i]).split()
    No_words[i] = len(Word_list)

Corr = np.hstack((Hospz,Retweet,No_words.reshape(-1,1)))

correlations = scipy.stats.spearmanr(Corr)
names = ["Class","Replies","Retweets","Likes","No_words",]
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations[0], vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,5,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)

df = pd.DataFrame(correlations[0]).to_csv('Correlation_Results_coeff.csv')
df = pd.DataFrame(correlations[1]).to_csv('Correlation_Results_p.csv')
plt.show()

### End of correlation analysis

print(Predictor.shape)
print(Hospz.shape)
print(Retweet.shape)

## Data processing for developing and validating the machine learning predictive models.

Predictor_semi = Predictor
Hospz_semi = Hospz
Vectors_semi = Vectors
Retweet_semi = Retweet
Date_semi = Date
No_words_semi = No_words


Days_test_1 = np.array(pd.date_range(start="2020-03-01",end="2020-03-31").to_pydatetime().tolist(),dtype='datetime64[D]').reshape(-1,1)
Days_test_2 = np.array(pd.date_range(start="2021-01-01",end="2021-01-31").to_pydatetime().tolist(),dtype='datetime64[D]').reshape(-1,1)
Days_test_3 = np.array(pd.date_range(start="2021-05-01",end="2021-05-31").to_pydatetime().tolist(),dtype='datetime64[D]').reshape(-1,1)
Days_test_4 = np.array(pd.date_range(start="2021-10-01",end="2021-10-31").to_pydatetime().tolist(),dtype='datetime64[D]').reshape(-1,1)

print("The 4 pandemic periods intervals and lengths")

print(np.unique(Date_semi[np.where(np.in1d(Date_semi,Days_test_1)==True)]))
print(len(No_words_semi[np.where(np.in1d(Date_semi,Days_test_1)==True)]))

print(np.unique(Date_semi[np.where(np.in1d(Date_semi,Days_test_2)==True)]))
print(len(No_words_semi[np.where(np.in1d(Date_semi,Days_test_2)==True)]))

print(np.unique(Date_semi[np.where(np.in1d(Date_semi,Days_test_3)==True)]))
print(len(No_words_semi[np.where(np.in1d(Date_semi,Days_test_3)==True)]))

print(np.unique(Date_semi[np.where(np.in1d(Date_semi,Days_test_4)==True)]))
print(len(No_words_semi[np.where(np.in1d(Date_semi,Days_test_4)==True)]))


Days_test = np.hstack((Days_test_1,Days_test_2,Days_test_3,Days_test_4))


for z in range (1,2):


    ## Cross-validation of the machine learning model

    print("Cross-validation begins")


    accuracy_final_rnn = np.zeros((1,))
    precision_final_rnn = np.zeros((1,))
    recall_final_rnn = np.zeros((1,))
    f1_final_rnn = np.zeros((1,))
    matthews_final_rnn = np.zeros((1,))
    roc_final_ovo_rnn = np.zeros((1,))
    roc_final_ovr_rnn = np.zeros((1,))

    for i in range (0,10):
        print(i)
        skf = StratifiedKFold(n_splits = 5, shuffle = True)
        skf.get_n_splits(Predictor, Hospz)
        for train_index, test_index in skf.split(Predictor, Hospz):
            Predictor_train, Predictor_test = Predictor[train_index], Predictor[test_index]
            Outcome_train, Outcome_test = Hospz[train_index], Hospz[test_index]

            
            X_train = Predictor_train.ravel()
            X_test = Predictor_test.ravel()
            y_train = Outcome_train.ravel()
            y_test = Outcome_test.ravel()

            # Clear layers and models created in the last session
            
            tf.keras.backend.clear_session()

            SEED = 42
            # Set random seed (assures reproducibility 
            random.seed(SEED)
            np.random.seed(SEED)
            tf.random.set_seed(SEED)

            # Process text in order to feed it to the RCNN model

            tokenizer = Tokenizer(num_words=1000)
            tokenizer.fit_on_texts(np.vstack((X_train.reshape(-1,1),X_test.reshape(-1,1))).ravel())
            X_train_seq = tokenizer.texts_to_sequences(X_train)
            X_test_seq = tokenizer.texts_to_sequences(X_test)
            max_len = 128
            X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)
            X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)

            # Define and train the RCNN model - the model is hybrid and contains layers corresponding both to convolutional and recurrent neural networks
            
            model = Sequential()
            model.add(Embedding(input_dim=1000, output_dim=128, input_length=max_len))
            model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(LSTM(units=64))
            model.add(Dense(units=3, activation='softmax'))
            model.compile(loss='sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate=0.005))
            model.fit(X_train_padded, y_train, batch_size=16, epochs=5)

            # Make predictions on the test set
            y_pred_prob = model.predict(X_test_padded)
            y_pred = np.argmax(y_pred_prob, axis=1)

            # Compute the ROC AUC score
            roc_auc_ovo = roc_auc_score(y_test, y_pred_prob, multi_class='ovo')
            roc_auc_ovr = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred,average = 'macro')
            recall = recall_score(y_test, y_pred,average = 'macro')
            f1 = f1_score(y_test, y_pred,average = 'macro')                    
            matthews = matthews_corrcoef(y_test, y_pred)

            print(f'OVO ROC AUC score: {roc_auc_ovo:.3f}')
            print(f'OVR ROC AUC score: {roc_auc_ovr:.3f}')

            print(f'Accuracy: {accuracy:.3f}')
            print(f'Precision: {precision:.3f}')
            print(f'Recall: {recall:.3f}')
            print(f'F1 score: {f1:.3f}')
            print(f'Matthews Correlation Coefficient: {matthews:.3f}')

            accuracy_final_rnn = np.vstack((accuracy_final_rnn,accuracy))
            precision_final_rnn = np.vstack((precision_final_rnn,precision))
            recall_final_rnn = np.vstack((recall_final_rnn,recall))
            f1_final_rnn = np.vstack((f1_final_rnn,f1))
            matthews_final_rnn = np.vstack((matthews_final_rnn,matthews))
            roc_final_ovo_rnn = np.vstack((roc_final_ovo_rnn,roc_auc_ovo))
            roc_final_ovr_rnn = np.vstack((roc_final_ovr_rnn,roc_auc_ovr))

    print("Average CV Accuracy Score")
    print(np.mean(accuracy_final_rnn[1:]))
                    
    print("Average CV Precision Score")
    print(np.mean(precision_final_rnn[1:]))
                    
    print("Average CV Recall Score")
    print(np.mean(recall_final_rnn[1:]))
                    
    print("Average CV F1 Score")
    print(np.mean(f1_final_rnn[1:]))
                    
    print("Average CV Matthews Correlation Coefficient")
    print(np.mean(matthews_final_rnn[1:]))
                    
    print("Average CV ROC AUC OVO Score")
    print(np.mean(roc_final_ovo_rnn[1:]))

    print("Average CV ROC AUC OVR Score")
    print(np.mean(roc_final_ovr_rnn[1:]))

    accuracy_final_final = np.array(np.mean(accuracy_final_rnn[1:])).reshape((1,1))
    precision_final_final = np.array(np.mean(precision_final_rnn[1:])).reshape((1,1))
    recall_final_final = np.array(np.mean(recall_final_rnn[1:])).reshape((1,1))
    f1_final_final = np.array(np.mean(f1_final_rnn[1:])).reshape((1,1))
    matthews_final_final = np.array(np.mean(matthews_final_rnn[1:])).reshape((1,1))
    roc_ovo_final_final = np.array(np.mean(roc_final_ovo_rnn[1:])).reshape((1,1))
    roc_ovr_final_final = np.array(np.mean(roc_final_ovr_rnn[1:])).reshape((1,1))

            
    a = pd.DataFrame((np.vstack((accuracy_final_final,precision_final_final,recall_final_final,f1_final_final,matthews_final_final,
                                 roc_ovo_final_final,roc_ovr_final_final))),
                     index = ['Accuracy','Precision','Recall','F1 Score','Matthews Correlation Coefficient','ROC AUC Score (OVO)','ROC AUC Score (OVR)']).to_csv('Internal_CV_Results_RCNN'+'.csv')


    ## External validation on 100 tweets from april 2021

    accuracy_final_rnn = np.zeros((1,))
    precision_final_rnn = np.zeros((1,))
    recall_final_rnn = np.zeros((1,))
    f1_final_rnn = np.zeros((1,))
    matthews_final_rnn = np.zeros((1,))
    roc_final_ovo_rnn = np.zeros((1,))
    roc_final_ovr_rnn = np.zeros((1,))

    print("External validation on 100 tweets from April 2021 begins")

    # Import the external validation data, along with the manual annotation of each of the 9 annotators.

    Class_1 = np.array(pd.read_excel("Data_inter-agree_100_final_1.xlsx",usecols = (12,),header = None,dtype = str,skiprows = 1),dtype=str).reshape(-1,1)
    Class_2 = np.array(pd.read_excel("Data_inter-agree_100_final_2.xlsx",usecols = (11,),header = None,dtype = str,skiprows = 1),dtype=str).reshape(-1,1)
    Class_3 = np.array(pd.read_excel("Data_inter-agree_100_final_3.xlsx",usecols = (11,),header = None,dtype = str,skiprows = 1),dtype=str).reshape(-1,1)
    Class_4 = np.array(pd.read_excel("Data_inter-agree_100_final_4.xlsx",usecols = (11,),header = None,dtype = str,skiprows = 1),dtype=str).reshape(-1,1)
    Class_5 = np.array(pd.read_excel("Data_inter-agree_100_final_5.xlsx",usecols = (11,),header = None,dtype = str,skiprows = 1),dtype=str).reshape(-1,1)
    Class_6 = np.array(pd.read_excel("Data_inter-agree_100_final_6.xlsx",usecols = (11,),header = None,dtype = str,skiprows = 1),dtype=str).reshape(-1,1)
    Class_7 = np.array(pd.read_excel("Data_inter-agree_100_final_7.xlsx",usecols = (11,),header = None,dtype = str,skiprows = 1),dtype=str).reshape(-1,1)
    Class_8 = np.array(pd.read_excel("Data_inter-agree_100_final_8.xlsx",usecols = (11,),header = None,dtype = str,skiprows = 1),dtype=str).reshape(-1,1)
    Class_9 = np.array(pd.read_excel("Data_inter-agree_100_final_9.xlsx",usecols = (11,),header = None,dtype = str,skiprows = 1),dtype=str).reshape(-1,1)

    Class_final = np.hstack((Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9))

    Class_final[Class_final=='True']='0'
    Class_final[Class_final=='NONE']='1'
    Class_final[Class_final=='FAKE']='2'

    Class_final = Class_final.astype(float)

    Majority = np.zeros((len(Class_1),))

    for i in range (0,len(Majority)):
        unique,counts = np.unique(Class_final[i,:],return_counts = True)
        if len(unique)==1:
            print("Perfect agreement")
        if len(unique[counts==np.max(counts)])==1:
            Majority[i] = float(unique[counts==np.max(counts)])
        elif len(unique[counts==np.max(counts)])==2:
            print(i)
            print(counts)
            Majority[i]=2
        else:
            Majority[i]=1

    Hospz_ext = Majority.reshape(-1,1)


    Predictor_ext = np.array(pd.read_excel("Data_inter-agree_100_final_1.xlsx",usecols = (3,),header = None,dtype = str,skiprows = 1),dtype=str).reshape(-1,1)

    Predictor_ext = np.hstack((Predictor_ext,Predictor_ext))

    # Preprocessing of the tweet content from the external dataset

    for i in range (0,len(Predictor_ext[:,0])):
        a = Predictor_ext[i,0]
        Predictor_ext[i,0] = " ".join(filter(lambda x:x[0]!='@', a.split()))

    for i in range (0,len(Predictor_ext[:,0])):
        a = Predictor_ext[i,0]
        Predictor_ext[i,0] = re.sub(r'http\S+', '',a)

    News = Predictor_ext[:,0]
    No_words_ext = np.zeros((len(News),))

    for i in range (0,len(News)):
        Word_list = str(News[i]).split()
        No_words_ext[i] = len(Word_list)

    text = Predictor_ext[:,0]
    Vectors_ext = np.zeros((len(text),96))
    for i in range (0,len(text)):
        text[i] = str(nlp(str(text[i])))
        a = nlp(str(text[i]))
        b = [word.lower_ for word in a]
        c = [ word for word in b if word not in STOP_WORDS and word not in punctuations ]
        Vectors_ext[i,:] = nlp(str(text[i])).vector
        Predictor_ext[i,0] = str(c)
        Predictor_ext[i,0] = unidecode.unidecode(Predictor_ext[i,0])

    Predictor_ext = Predictor_ext[:,0]

    print(Predictor_semi.shape)
    print(Predictor_ext.shape)

    X_train = Predictor_semi.ravel()
    X_test = Predictor_ext.ravel()
    y_train = Hospz_semi.ravel()
    y_test = Hospz_ext.ravel()

    # Clear layers and models created in the last session
            
    tf.keras.backend.clear_session()

    SEED = 42
    # Set random seed (assures reproducibility 
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Process text in order to feed it to the RCNN model

    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(np.vstack((X_train.reshape(-1,1),X_test.reshape(-1,1))).ravel())
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    max_len = 128
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)

    # Define and train the RCNN model - the model is hybrid and contains layers corresponding both to convolutional and recurrent neural networks
            
    model = Sequential()
    model.add(Embedding(input_dim=1000, output_dim=128, input_length=max_len))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=64))
    model.add(Dense(units=3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate=0.005))
    model.fit(X_train_padded, y_train, batch_size=16, epochs=5)

    # Make predictions on the test set
    y_pred_prob = model.predict(X_test_padded)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Compute the ROC AUC score
    roc_auc_ovo = roc_auc_score(y_test, y_pred_prob, multi_class='ovo')
    roc_auc_ovr = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average = 'macro')
    recall = recall_score(y_test, y_pred,average = 'macro')
    f1 = f1_score(y_test, y_pred,average = 'macro')                    
    matthews = matthews_corrcoef(y_test, y_pred)

    print(f'OVO ROC AUC score: {roc_auc_ovo:.3f}')
    print(f'OVR ROC AUC score: {roc_auc_ovr:.3f}')

    print(f'Accuracy: {accuracy:.3f}')
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 score: {f1:.3f}')
    print(f'Matthews Correlation Coefficient: {matthews:.3f}')

    # Create the boxplot representation of the predicted probabilities

    test_labels_int = np.array(y_test.ravel(),dtype = float)
    prediction_proba = y_pred_prob


    a = pd.DataFrame(np.hstack((np.vstack((prediction_proba[:,0].reshape(-1,1),prediction_proba[:,1].reshape(-1,1),prediction_proba[:,2].reshape(-1,1))),np.vstack((test_labels_int.reshape(-1,1),test_labels_int.reshape(-1,1),test_labels_int.reshape(-1,1))),
                                    np.vstack((np.zeros(test_labels_int.shape).reshape(-1,1),np.ones(test_labels_int.shape).reshape(-1,1),(2*np.ones(test_labels_int.shape).reshape(-1,1)))))),columns = ['Predicted probabilities','Annotated class','Class probabilities'])

    a['Annotated class'] = a['Annotated class'].replace(0,'True')
    a['Annotated class'] = a['Annotated class'].replace(1,'Neutral')
    a['Annotated class'] = a['Annotated class'].replace(2,'Fake')

    a['Class probabilities'] = a['Class probabilities'].replace(0,'True')
    a['Class probabilities'] = a['Class probabilities'].replace(1,'Neutral')
    a['Class probabilities'] = a['Class probabilities'].replace(2,'Fake')
            
    sns.set_style('whitegrid')
    my_pal = {"True": "b", "Neutral": "m", "Fake":"r"}
    ax = sns.boxplot(x='Annotated class',y='Predicted probabilities',hue = 'Class probabilities',data=a, palette = my_pal)
    plt.show()


    accuracy_final_rnn = np.vstack((accuracy_final_rnn,accuracy))
    precision_final_rnn = np.vstack((precision_final_rnn,precision))
    recall_final_rnn = np.vstack((recall_final_rnn,recall))
    f1_final_rnn = np.vstack((f1_final_rnn,f1))
    matthews_final_rnn = np.vstack((matthews_final_rnn,matthews))
    roc_final_ovo_rnn = np.vstack((roc_final_ovo_rnn,roc_auc_ovo))
    roc_final_ovr_rnn = np.vstack((roc_final_ovr_rnn,roc_auc_ovr))

    print("External validation Accuracy Score")
    print(np.mean(accuracy_final_rnn[1:]))
                    
    print("External validation Precision Score")
    print(np.mean(precision_final_rnn[1:]))
                    
    print("External validation Recall Score")
    print(np.mean(recall_final_rnn[1:]))
                    
    print("External validation F1 Score")
    print(np.mean(f1_final_rnn[1:]))
                    
    print("External validation Matthews Correlation Coefficient")
    print(np.mean(matthews_final_rnn[1:]))
                    
    print("External validation ROC AUC OVO Score")
    print(np.mean(roc_final_ovo_rnn[1:]))

    print("External validation ROC AUC OVR Score")
    print(np.mean(roc_final_ovr_rnn[1:]))

    accuracy_final_final = np.array(np.mean(accuracy_final_rnn[1:])).reshape((1,1))
    precision_final_final = np.array(np.mean(precision_final_rnn[1:])).reshape((1,1))
    recall_final_final = np.array(np.mean(recall_final_rnn[1:])).reshape((1,1))
    f1_final_final = np.array(np.mean(f1_final_rnn[1:])).reshape((1,1))
    matthews_final_final = np.array(np.mean(matthews_final_rnn[1:])).reshape((1,1))
    roc_ovo_final_final = np.array(np.mean(roc_final_ovo_rnn[1:])).reshape((1,1))
    roc_ovr_final_final = np.array(np.mean(roc_final_ovr_rnn[1:])).reshape((1,1))

            
    a = pd.DataFrame((np.vstack((accuracy_final_final,precision_final_final,recall_final_final,f1_final_final,matthews_final_final,
                                 roc_ovo_final_final,roc_ovr_final_final))),
                     index = ['Accuracy','Precision','Recall','F1 Score','Matthews Correlation Coefficient','ROC AUC Score (OVO)','ROC AUC Score (OVR)']).to_csv('External_Results_RCNN'+'.csv')

    ## Internal validation based on the 4 pandemic periods.

    print("Internal validation based on the 4 pandemic periods begins")

    accuracy_final_rnn = np.zeros((1,))
    precision_final_rnn = np.zeros((1,))
    recall_final_rnn = np.zeros((1,))
    f1_final_rnn = np.zeros((1,))
    matthews_final_rnn = np.zeros((1,))
    roc_final_ovo_rnn = np.zeros((1,))
    roc_final_ovr_rnn = np.zeros((1,))

    for k in range (0,5):
       
        for j in range (0,4):

            Predictor = Predictor_semi[np.where(np.in1d(Date_semi,Days_test[:,j])==False)]
            Predictor_test1 = Predictor_semi[np.where(np.in1d(Date_semi,Days_test[:,j])==True)]

            Hospz = Hospz_semi[np.where(np.in1d(Date_semi,Days_test[:,j])==False)]
            Hospz_test1 = Hospz_semi[np.where(np.in1d(Date_semi,Days_test[:,j])==True)]

            X_train = Predictor.ravel()
            X_test = Predictor_test1.ravel()
            y_train = Hospz.ravel()
            y_test = Hospz_test1.ravel()

            # Clear layers and models created in the last session
                    
            tf.keras.backend.clear_session()

            SEED = 42
            # Set random seed (assures reproducibility 
            random.seed(SEED)
            np.random.seed(SEED)
            tf.random.set_seed(SEED)

            # Process text in order to feed it to the RCNN model

            tokenizer = Tokenizer(num_words=1000)
            tokenizer.fit_on_texts(np.vstack((X_train.reshape(-1,1),X_test.reshape(-1,1))).ravel())
            X_train_seq = tokenizer.texts_to_sequences(X_train)
            X_test_seq = tokenizer.texts_to_sequences(X_test)
            max_len = 128
            X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)
            X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)

            # Define and train the RCNN model - the model is hybrid and contains layers corresponding both to convolutional and recurrent neural networks
                    
            model = Sequential()
            model.add(Embedding(input_dim=1000, output_dim=128, input_length=max_len))
            model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(LSTM(units=64))
            model.add(Dense(units=3, activation='softmax'))
            model.compile(loss='sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate=0.005))
            model.fit(X_train_padded, y_train, batch_size=16, epochs=5)

            # Make predictions on the test set
            y_pred_prob = model.predict(X_test_padded)
            y_pred = np.argmax(y_pred_prob, axis=1)

            # Compute the ROC AUC score
            roc_auc_ovo = roc_auc_score(y_test, y_pred_prob, multi_class='ovo')
            roc_auc_ovr = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred,average = 'macro')
            recall = recall_score(y_test, y_pred,average = 'macro')
            f1 = f1_score(y_test, y_pred,average = 'macro')                    
            matthews = matthews_corrcoef(y_test, y_pred)

            print(f'OVO ROC AUC score: {roc_auc_ovo:.3f}')
            print(f'OVR ROC AUC score: {roc_auc_ovr:.3f}')

            print(f'Accuracy: {accuracy:.3f}')
            print(f'Precision: {precision:.3f}')
            print(f'Recall: {recall:.3f}')
            print(f'F1 score: {f1:.3f}')
            print(f'Matthews Correlation Coefficient: {matthews:.3f}')

            accuracy_final_rnn = np.vstack((accuracy_final_rnn,accuracy))
            precision_final_rnn = np.vstack((precision_final_rnn,precision))
            recall_final_rnn = np.vstack((recall_final_rnn,recall))
            f1_final_rnn = np.vstack((f1_final_rnn,f1))
            matthews_final_rnn = np.vstack((matthews_final_rnn,matthews))
            roc_final_ovo_rnn = np.vstack((roc_final_ovo_rnn,roc_auc_ovo))
            roc_final_ovr_rnn = np.vstack((roc_final_ovr_rnn,roc_auc_ovr))

    accuracy_final_final = np.array(np.mean(accuracy_final_rnn[1:])).reshape((1,1))
    precision_final_final = np.array(np.mean(precision_final_rnn[1:])).reshape((1,1))
    recall_final_final = np.array(np.mean(recall_final_rnn[1:])).reshape((1,1))
    f1_final_final = np.array(np.mean(f1_final_rnn[1:])).reshape((1,1))
    matthews_final_final = np.array(np.mean(matthews_final_rnn[1:])).reshape((1,1))
    roc_ovo_final_final = np.array(np.mean(roc_final_ovo_rnn[1:])).reshape((1,1))
    roc_ovr_final_final = np.array(np.mean(roc_final_ovr_rnn[1:])).reshape((1,1))

    a = pd.DataFrame((np.vstack((accuracy_final_final,precision_final_final,recall_final_final,f1_final_final,matthews_final_final,
                                roc_ovo_final_final,roc_ovr_final_final))),
                    index = ['Accuracy','Precision','Recall','F1 Score','Matthews Correlation Coefficient','ROC AUC Score (OVO)','ROC AUC Score (OVR)']).to_csv('Internal_Period_Results_RCNN'+'.csv')

    
    

            
