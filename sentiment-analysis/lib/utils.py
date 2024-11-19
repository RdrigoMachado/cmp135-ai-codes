import re, os
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd


#!pip install nltk
#!python3 -m nltk.downloader stopwords
from nltk.corpus import stopwords




def remove_emoji(string):
  emoji_pattern = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          "]+", flags=re.UNICODE)
  return emoji_pattern.sub(r'', string)

# Method for clearing strings
# Removes non-meaningful content
def clean_str(string):
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  string = re.sub(r'RT+', '', string)
  string = re.sub(r'@\S+', '', string)
  string = re.sub(r'http\S+', '', string)
  
  cleanr = re.compile('<.*?>')
  string = re.sub(r'\d+', '', string)
  string = re.sub(cleanr, '', string)
  string = re.sub("'", '', string)
  string = re.sub(r'\W+', ' ', string)
  
  string = string.replace('_', '')
  
  string = remove_emoji(string)
  
  return string.strip().lower()



def prepare_data(data, max_features, maxlen, test_dim):
  data = data[['text', 'sentiment']]
  data['text'] = data['text'].apply(lambda x: x.lower())
  data['text'] = data['text'].apply(lambda x: clean_str(x))
  data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

  stop_words = set(stopwords.words('english'))
  text = []
  for row in data['text'].values:
    word_list = text_to_word_sequence(row)
    no_stop_words = [w for w in word_list if not w in stop_words]
    no_stop_words = " ".join(no_stop_words)
    text.append(no_stop_words)

  tokenizer = Tokenizer(num_words=max_features, split=' ')

  tokenizer.fit_on_texts(text)
  X = tokenizer.texts_to_sequences(text)

  X = pad_sequences(X, maxlen=maxlen)

  word_index = tokenizer.word_index
  Y = pd.get_dummies(data['sentiment']).values
  # Split the dataset for training and testing
  X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_dim, random_state=42, stratify=Y)
  
  X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 41, stratify=Y_train)
  return X_train, X_val, X_test, Y_train, Y_val, Y_test, word_index, tokenizer
