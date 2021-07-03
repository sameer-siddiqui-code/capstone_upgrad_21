import numpy as np
from flask import Flask, render_template, request
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
ps = PorterStemmer()
#ensuring that the code does not break if the corpus is not downlaoded already
nltk.download('stopwords')
stopword=set(stopwords.words('english'))

app = Flask(__name__)

#text cleaning function
def text_process(text):
    text = text.lower()
    text = re.sub('[^a-z]+', ' ', text).strip()
    text = text.split()
    text = [ word for word in text if not word in stopword ]
    text = [ ps.stem(word) for word in text ]
    return ' '.join(text)


#loading models plus reviews-product mapping file
sentimental_model = pickle.load(open('finalized_model_sentiment.pkl','rb'))
recommendation_model=pd.read_csv('user-user collabrative filtering model.csv',index_col = 'reviews_username')
product_reviews = pd.read_csv("product_reviews.csv")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=("POST", "GET"))
def predict():
    user = [ str(x) for x in request.form.values() ][0]
    #user = request.form.values()
    recommended_products=recommendation_model.loc[user].sort_values(ascending=False)[0:20].index.to_list()
    product_reviews_recommended=product_reviews[product_reviews['name'].isin(recommended_products)]
   
    X = product_reviews_recommended['reviews_text']
    X = X.apply(lambda x: text_process(x))
    from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word', stop_words= 'english')
    X_features = tfidf.fit_transform(X)
    y=sentimental_model.predict(X_features)
    product_reviews_recommended['sentiment']=y
    product_reviews_recommended['sentiment']=product_reviews_recommended['sentiment'].apply(lambda x:"Negative" if x==0 else "Positive")
    product_reviews_recommended=pd.crosstab(product_reviews_recommended.name,product_reviews_recommended.sentiment)
    product_reviews_recommended['Positive_Ratio']=product_reviews_recommended['Positive']/(product_reviews_recommended['Positive']+product_reviews_recommended['Negative'])
    product_reviews_recommended=product_reviews_recommended.sort_values('Positive_Ratio',ascending=False)[0:5]
    product_reviews_recommended.reset_index()
    final=pd.DataFrame()
    final['Recommended_Products']=product_reviews_recommended.index
    
    
    
    return render_template('predict.html', prediction_text= 'Top recommended products for user {} are '.format(user),tables=[final.to_html(classes='center',index=False,justify='center')], titles=final.columns.values)
    #return render_template('index.html', prediction_text= 'Top recommended products for user {} are :'.format(user))


if __name__=='__main__':
    app.run(debug=True)