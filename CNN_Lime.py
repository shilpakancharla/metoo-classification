#!/usr/bin/env python
# coding: utf-8

# # Modeling
# 
# There are three steps to creating this model:
# 
# 1. **Vectorization**
# 2. **Train/Validation/Test Split**
# 3. **Modeling**: We apply a baseline CNN, FastText, and LSTM models.

# In[65]:


import numpy as np
import re
import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import layers
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Conv1D, LSTM, GlobalMaxPooling1D, InputLayer, Dropout, SpatialDropout1D
from keras import models
from keras import losses
from keras import metrics
from lime.lime_text import LimeTextExplainer
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use(style = 'seaborn')

#import sys
#!{sys.executable} -m pip install lime


# In[23]:


# Import the dataset
df = pd.read_csv('processed_data/clean_data.csv')
df = df.dropna()


# In[45]:


print(df['Years'].unique())


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(df['Tweets with no Stopwords'], 
                                                    df['Years'], test_size = 0.3, random_state = 2)


# In[25]:


token = Tokenizer(num_words = 5000, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n', lower = True, 
                  split = ' ', oov_token = True)
token.fit_on_texts(X_train)
X_train_seq = token.texts_to_sequences(X_train)
X_test_seq = token.texts_to_sequences(X_test)


# In[26]:


# One hot encoding for labels
encoder = LabelBinarizer()
encoder.fit(y_train)
transformed = encoder.transform(y_train)
y_train_encoded = pd.DataFrame(transformed)
transformed_test = encoder.transform(y_test)
y_test_encoded = pd.DataFrame(transformed_test)


# In[28]:


def max_seq_length(sequence):
    length = []
    for i in range(0, len(sequence)):
        length.append(len(sequence[i]))
    return max(length)

max_length = max_seq_length(X_train_seq)
# Total number of words in the corpus
vocabulary_size = len(token.word_index)


# In[29]:


X_train_seq_pad = pad_sequences(X_train_seq, maxlen = max_length, padding = 'post')
X_test_seq_pad = pad_sequences(X_test_seq, maxlen = max_length, padding = 'post')


# In[30]:


X_train_emb, X_val_emb, y_train_emb, y_val_emb = train_test_split(X_train_seq_pad, y_train_encoded, 
                                                                  test_size = 0.3, random_state = 3)


# # CNN for Text Analysis
# 
# Neural networks analyze texts in a slightly different way with words as opposed to the sparse TF-IDF framework. Since this is a large dataset, a CNN maybe able to pick up intricate patterns. Preprocessing with CNNs requires it to be processed with Keras' `Embedding.()` when it comes to the modeling.

# In[52]:


def define_CNN_model():
    cnn_model = models.Sequential()
    cnn_model.add(layers.Embedding(input_dim = vocabulary_size + 1, output_dim = 200, input_length = max_length))
    cnn_model.add(layers.Conv1D(50, 3, activation = 'relu', input_shape = (200, 1)))
    cnn_model.add(layers.GlobalMaxPooling1D())
    cnn_model.add(Dropout(0.3))
    cnn_model.add(layers.Dense(3, activation = 'softmax'))
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return cnn_model


# In[53]:


cnn_model = define_CNN_model()
cnn_model.summary()
cnn_model_history = cnn_model.fit(X_train_emb, y_train_emb, epochs = 10, batch_size = 64, 
                                  validation_data = (X_val_emb, y_val_emb), verbose = 1)


# In[59]:


def plot_history(history):
    # Plot loss
    plt.title('Loss')
    plt.plot(history.history['loss'], color = 'blue', label = 'train')
    plt.plot(history.history['val_loss'], color = 'red', label = 'test')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()

    # Plot accuracy
    plt.title('Accuracy')
    plt.plot(history.history['acc'], color = 'blue', label = 'train')
    plt.plot(history.history['val_acc'], color = 'red', label = 'test')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()

plot_history(cnn_model_history)


# In[55]:


# Get classification report
y_pred = cnn_model.predict(X_test_seq_pad, batch_size = 64, verbose = 1)
y_pred_bool = encoder.inverse_transform(y_pred) # Undo one-hot encoding
print(classification_report(y_test, y_pred_bool))


# In[57]:


# Save this model
# serialize model to JSON
model_json = cnn_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn_model.save_weights("model.h5")
print("Saved model to disk")


# In[77]:


# GET the probability for each category
y_pred_2017 = y_pred[:, 0]
y_pred_2018 = y_pred[:, 1]
y_pred_2019 = y_pred[:, 2]
# give them index
ind_2017 = np.argsort(y_pred_2017)
ind_2018 = np.argsort(y_pred_2018)
ind_2019 = np.argsort(y_pred_2019)
print(ind_2017)
num = 20
print(y_pred_2017[ind_2017[-1:-num:-1]])

# get a explainer object
explainer = LimeTextExplainer(class_names = {'2017', '2018', '2019'})
# define a new function:
def new_predict(texts):
    _seq = token.texts_to_sequences(texts)
    _pad = pad_sequences(_seq, maxlen = max_length, padding = 'post')
    return cnn_model.predict(_pad)
    #return _seq, _pad, cnn_model.predict(_pad)
for ii in X_test.iloc[ind_2019[-1:-num:-1]]:
    exp = explainer.explain_instance(ii, new_predict, num_features = max_length, top_labels = 1)
    exp.show_in_notebook(text = True)


# In[78]:


for ii in X_test.iloc[ind_2017[-1:-num:-1]]:
    exp = explainer.explain_instance(ii, new_predict, num_features = max_length, top_labels = 1)
    exp.show_in_notebook(text = True)


# In[79]:


for ii in X_test.iloc[ind_2018[-1:-num:-1]]:
    exp = explainer.explain_instance(ii, new_predict, num_features = max_length, top_labels = 1)
    exp.show_in_notebook(text = True)

