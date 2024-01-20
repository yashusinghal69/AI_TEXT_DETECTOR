from flask import Flask, request, jsonify
from flask_cors import CORS   
import pickle

app = Flask(__name__)
CORS(app)  

with open('vectorizer.pkl', 'rb') as f:
    to_vector = pickle.load(f)

with open('model_nb.pkl', 'rb') as f:
    model_new = pickle.load(f)
    
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Initialize Porter Stemmer and stopwords
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Define the preprocess function
def remove_stop_words(sentence):
    words = sentence.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def remove_html(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)

def remove_punct(data):
    return data.translate(str.maketrans('', '', string.punctuation))

def remove_url(data):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', data)

def stem_sentence(sentence):
    tokens = word_tokenize(sentence)
    stemmed_tokens = [porter.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def preprocess(data):
    data = data.lower()
    data = remove_html(data)
    data = remove_url(data)
    data = remove_punct(data)
    data = remove_stop_words(data)
    data = stem_sentence(data)
    return data


@app.route('/',)
def test():
    return 'Welcome!'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_sms = data['text']

    if input_sms is not None:
        transformed_sms = preprocess(input_sms)
        vector_input = to_vector.transform([transformed_sms])
        result = model_new.predict_proba(vector_input)[0]

        human = round(result[0] * 100,1)
        ai = round(result[1] * 100,1)

        response = {
            "probabilities": {
                "ai": ai,
                "human": human
            }
        }

        return jsonify(response)
    else:
        return jsonify({'error': 'Input text not provided.'})
    
if __name__ == '__main__':
    app.run(debug=True)
        
