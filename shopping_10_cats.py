import streamlit as st
import pandas as pd
import requests
import random
from hashlib import md5
import jieba
import re
import pickle
import keras
from keras.preprocessing.sequence import pad_sequences

cat_id_df = pd.read_csv('shopping_10_cats/cat_id.csv')
classification_report = pd.read_csv('shopping_10_cats/classification_report.csv')
df_cat_count = pd.read_csv('shopping_10_cats/df_cat_count.csv')
Indicator_cat10 = pd.read_csv('shopping_10_cats/Indicator_cat10.csv')

st.title('Training data：')
st.write(
    "Data overview: 10 categories (books, tablets, mobile phones, fruits, shampoos, water heaters, Mengniu, clothes, "
    "computers, hotels), with a total of more than 60000 comment data, with about 30000 positive and negative "
    "comments each")
st.dataframe(df_cat_count)

# st.title('Model weight:')
# st.image('shopping_10_cats/shopping_10_cats.png', channels="BGR")

# st.title('Model evaluation indicators:')
# st.write('Accuracy')
# st.line_chart(Indicator_cat10[["acc", "val_acc"]])
# st.write('loss')
# st.line_chart(Indicator_cat10[["loss", "val_loss"]])

st.title('Classification report:')
st.dataframe(classification_report)


def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line


def stopwords_list(filepath):
    stopword = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopword


# Load stop words
stopwords = stopwords_list("stopwords.txt")

# loading tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
MAX_SEQUENCE_LENGTH = 250
model = keras.models.load_model('models/shopping_10_cats.h5')


def translate(string):
    # Set your own appid/appkey.
    appid = '20220911001339781'
    appkey = 'oGkv8qDxc1lQjeiA0nQW'

    # For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
    from_lang = 'en'
    to_lang = 'zh'

    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path

    query = string

    # Generate salt and sign
    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()
    return result["trans_result"][0]["dst"]


def predict(text):
    txt = remove_punctuation(text)
    txt = [" ".join([w for w in list(jieba.cut(txt)) if w not in stopwords])]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    cat_id = pred.argmax(axis=1)[0]
    return cat_id_df[cat_id_df.cat_id == cat_id]['cat'].values[0]


st.title('Forecast:')

text_input = st.text_input(
    "Enter a product comment 👇",
)

if text_input:
    st.write("You entered: ", text_input)
    pred = predict(translate(text_input))
    st.caption('The predicted result is: ')
    st.title(pred)
