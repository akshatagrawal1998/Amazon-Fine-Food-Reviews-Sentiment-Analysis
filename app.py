from flask import Flask, render_template, request
import pickle 

import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

model = pickle.load(open(r'C:\Users\akshat.agrawal\Desktop\ML models\Amazon Fine Food Reviews Sentimental Analysis\Amazon Model Deployment\amazon_review_model.pkl'
                         , 'rb'))
vectorizer = pickle.load(open(r'C:\Users\akshat.agrawal\Desktop\ML models\Amazon Fine Food Reviews Sentimental Analysis\Amazon Model Deployment\vectorizer.pkl','rb'))
#prediction

#review = 'I liked my laptop very much but its too expensive'
#sent = pd.Series(review)

#sent1 = vectorizer.transform(sent)
app = Flask(__name__)

@app.route("/")
def func():
    return render_template("amazon_form.html")

@app.route("/sentiment" , methods = ['POST'])
def get_sentiment():
    review = request.form.get('statement')
    #sent = pd.Series(review)
    #sent1 = vectorizer.transform(sent).reshape(-1,1)
    
    # 1. preprocess
    transformed_review = transform_text(review)
    # 2. vectorize
    vector_input = vectorizer.transform([transformed_review])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        return "Not Positive Review"
    else:
        return "Positive"

#result = model.predict(sent1)
#return str(result)

if __name__ == "__main__":
    app.run(debug=True)