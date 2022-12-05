from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
import sqlite3

import pickle 
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
    'title': LazyString(lambda:'Dokumentasi API'),
    'version': LazyString(lambda:'1.0.0'),
    'description': LazyString(lambda:'API Deep learning'),
    },
    host = LazyString(lambda: request.host)
)

swagger_config = {
    "headers": [],
    "specs":[
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs"
}

swagger = Swagger(app, template=swagger_template,config=swagger_config)

#INISIALISASI DATABASE
db = sqlite3.connect('storage.db',check_same_thread=False)
db.row_factory = sqlite3.Row
c = db.cursor()
c.execute("create table if not exists databases (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, sentiment TEXT);")
db.commit() 

#HOMEPAGE
@swag_from("docs/hello.yml", methods = ['GET'])
@app.route('/', methods = ['GET'])
def hello_world(): 
    json_response = {
        'status code': 200,
        'description' : 'Home',
        'data': 'Tweet and Word Cleaning',
    }
    response_data = jsonify(json_response)
    return response_data

#LSTM
max_features = 100000
tokenizer = pickle.load(open('/Users/muhammadaldan/projects/platinum/balancedlstm/tokenizer.pickle','rb'))
sentiment = ['negative', 'neutral', 'positive']


#MLPC
import pandas as pd

df= pd.read_csv('coba1.tsv', sep='\t', header=None)

df.head()

df.shape

df.columns =['Text', 'Class']
df

df.Class.value_counts()

import re 

def lowercase(text):
    return text.lower()

def remove_unnecessary_char(text):
    text = re.sub('\n',' ',text) 
    text = re.sub('rt',' ',text) 
    text = re.sub('user',' ',text) 
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text) 
    text = re.sub('  +', ' ', text) 
    return text
    
def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text) 
    return text

alay_dict = pd.read_csv('new_kamusalay.csv', encoding='latin-1', header=None)

alay_dict = alay_dict.rename(columns={0:'original',
                                    1:'replacement'})


alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
def baku(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])


def preprocess(text):
    text = lowercase(text) # 1
    text = remove_nonaplhanumeric(text) # 2
    text = remove_unnecessary_char(text) # 2
    text = baku(text) # 3
    return text

df['text_clean'] = df.Text.apply(preprocess)

df.head()


data_preprocessed = df.text_clean.tolist()

data_preprocessed


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(data_preprocessed)

X = tfidf_vect.transform(data_preprocessed)


#lstm
file = open('/Users/muhammadaldan/projects/platinum/balancedlstm/x_pad_sequences.pickle','rb')
feature_file_from_lstm = pickle.load(file)
file.close()

model_file_from_lstm = load_model('/Users/muhammadaldan/projects/platinum/balancedlstm/model.h5')

#mlpc
file = open('/Users/muhammadaldan/projects/platinum/balancedmlp/feature.pickle','rb')
feature_file_from_mlpc = pickle.load(file)
file.close()

model_file_from_mlpc = pickle.load(open('/Users/muhammadaldan/projects/platinum/balancedmlp/model.pickle','rb'))



#Databases
#CHECKING DATABASES
@swag_from("docs/check_db.yml", methods = ['GET'])
@app.route("/databases", methods = ['GET'])
def database_check():
    query = "select * from databases"
    select_tweet = c.execute(query)
    tweet = [
        dict(id=row[0], text=row[1], sentiment=row[2])
        for row in select_tweet.fetchall()
    ]
    

    json_response = {
        'status code': 200,
        'description' : 'Checking Database',
        'data': tweet,
    }
    response_data = jsonify(json_response)
    return response_data

#Cleaning Databases Data
@swag_from("docs/delete.yml",methods=['DELETE'])
@app.route("/delete_data", methods=['DELETE'])
def del_data():
    c.execute("DELETE FROM databases;")
    c.execute("delete from sqlite_sequence where name='databases';")
    db.commit()

    json_response = {
        'status code': 200,
        'description': 'Cleaning Databases Record',
        'data': 'Table Cleansed',
    }
    response_data = jsonify(json_response)
    return response_data

@swag_from("docs/lstm.yml", methods=['POST'])
@app.route("/lstm", methods=['POST'])
def lstm():
    original_text = request.form.get('text')
    text = [preprocess(original_text)]
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code': 200,
        'description': "Result of sentiment analysis",
        'data': {
            'text': text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

@swag_from("docs/mlpc.yml", methods=['POST'])
@app.route("/mlpc", methods=['POST'])
def mlpc():

    original_text = request.form.get('text')
    text = tfidf_vect.transform([preprocess(original_text)])
    result = model_file_from_mlpc.predict(text)[0]

    json_response = {
        'status_code': 200,
        'description': "Result of sentiment analysis",
        'data': {
            'text': original_text,
            'sentiment': result
        },
    }
    response_data = jsonify(json_response)
    return response_data

#INPUT CSV
@swag_from("docs/lstm_file.yml",methods=['POST'])
@app.route("/lstm-file", methods=['POST'])
def lstm_file():
    file = request.files["file"]
    try:
        df = pd.read_csv(file, sep='\t', header=None)
    except:
        df = pd.read_csv(file, encoding='utf-8')
    col_1 = df.iloc[:,0]
    for text in col_1:
        bersih = preprocess(text)
        pred = [preprocess(text)]
        feature = tokenizer.texts_to_sequences(pred)
        feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
        prediction = model_file_from_lstm.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]


        query = "insert into databases (text,sentiment) values(? , ?)"
        values = (bersih, get_sentiment)
        c.execute(query, values)
        db.commit()
    json_response = {
        'status code': 200,
        'description': 'CSV Upload and Processing',
        'data': 'File successfully uploaded',
    }
    response_data = jsonify(json_response)
    return response_data


@swag_from("docs/mlpc_file.yml",methods=['POST'])
@app.route("/mlpc-file", methods=['POST'])
def mlpc_file():
    file = request.files["file"]
    try:
        df = pd.read_csv(file, sep='\t', header=None)
    except:
        df = pd.read_csv(file, encoding='utf-8')
    col_1 = df.iloc[:,0]
    for text in col_1:
        bersih = preprocess(text)
        pred = [preprocess(text)]
        guess = tfidf_vect.transform(pred)
        result = model_file_from_mlpc.predict(guess)[0]


        query = "insert into databases (text,sentiment) values(? , ?)"
        values = (bersih, result)
        c.execute(query, values)
        db.commit()
    json_response = {
        'status code': 200,
        'description': 'CSV Upload and Processing',
        'data': 'File successfully uploaded',
    }
    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run()  