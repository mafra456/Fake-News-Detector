"""
@author: Andre Correia Lacerda Mafra
@Github: https://github.com/mafra456
"""

#Data Preprocessing

#importing the libraries
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import pandas as pd
import string
   
ps = PorterStemmer()
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "’", "‘"] + list(ENGLISH_STOP_WORDS))
PUNCTUATION = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "“", "”"]

#importing the dataset
dataset = pd.read_csv('dataset/fake_or_real_news.csv')

dataset['title'] = dataset['title'].str.lower()
dataset['title'] = dataset['title'].apply(nltk.word_tokenize)
dataset['text']  = dataset['text'].str.lower()
dataset['text']  = dataset['text'].apply(nltk.word_tokenize)

#Set the title tuple to be tokenized
tuples_title = [tuple(x) for x in dataset['title'].values]
corpus = []
df= []

for title in dataset["title"]:
    wtlt= " "
    for word in title:
        if word not in STOPLIST and word not in PUNCTUATION:
            word = ps.stem(word)
            wtlt += word
            wtlt += " "
    corpus.append(wtlt)

#Set the text tuple to be tokenized
tuples_text = [tuple(x) for x in dataset['text'].values]
i = 0
for text in dataset["text"]:
    wtxt=[]
    for word in text:
        if word not in STOPLIST and word not in PUNCTUATION:
            word = ps.stem(word)
            corpus[i] += word
            corpus[i] += " "
    i = i + 1            

df = pd.Series(corpus)

#transforming labels into numbers
dataset.loc[dataset["label"]=='FAKE',"label",]=1
dataset.loc[dataset["label"]=='REAL',"label",]=0

#X is data and y is label
X = df
y = dataset["label"]

#Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#tf-idf   
vectorizer = TfidfVectorizer(max_df = 0.8, min_df = 0.2) #Read about max_df and min_df
X_train_transformed = vectorizer.fit_transform(X_train).toarray()
X_test_transformed=vectorizer.transform(X_test).toarray()

y_train=y_train.astype('int')
y_test=y_test.astype('int')

#Train the SVM
clf = svm.SVC()
clf.fit(X_train_transformed, y_train)  
svm.SVC(kernel = 'linear', probability = True, random_state = 0)


#Train the NB
clf_mnb = MultinomialNB()
clf_mnb.fit(X_train_transformed, y_train)
predictions_mnb = clf_mnb.predict(X_test_transformed) 
#predict
predictions = clf.predict(X_test_transformed)
predictions_mnb = clf_mnb.predict(X_test_transformed) 
c = clf_mnb.predict_proba(X_test_transformed)
a=np.array(y_test)
b = np.array(predictions_mnb)
b.reshape(1267,1)
d = np.array(predictions)
d.reshape(1267,1)
#accuracy
count = 0
for i in range (len(predictions)):
    if predictions[i]==a[i]:
        count=count+1
        
print(count/len(predictions))
count = 0
for i in range (len(predictions_mnb)):
    if predictions_mnb[i]==a[i]:
        count=count+1
        
print(count/len(predictions_mnb))

#Precision and Recall for SVM with Tf
print(precision_score(a, d, average='macro'))
print(recall_score(a, d, average='macro')) 
#Precision and Recall for SVM with Tf-idf
#print(precision_score(a, d, average='macro'))
#print(recall_score(a, d, average='macro')) 
#Precision and Recall for MNB with Tf
print(precision_score(a, b, average='macro'))
print(recall_score(a, b, average='macro')) 
#Precision and Recall for MNB with Tf-idf
#print(precision_score(a, b, average='macro'))
#print(recall_score(a, b, average='macro')) 


print(f1_score(a, b, average='micro')) 
print(f1_score(a, d, average='micro')) 
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[0], tpr[0], threshold = roc_curve(a,c[:,0], pos_label=0)
fpr[1], tpr[1], threshold = roc_curve(a,c[:,1], pos_label=1)
roc_auc[0] = auc(fpr[0], tpr[0])
roc_auc[1] = auc(fpr[1], tpr[1])

plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot(fpr[1], tpr[1], color='aqua',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

