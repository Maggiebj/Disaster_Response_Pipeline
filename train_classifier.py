import sys
from sqlalchemy import create_engine
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report



class StartingVerbExtractor(BaseEstimator,TransformerMixin):   
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags)>0:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] :
                    return 1
                else: 
                    return 0
            else: 
                return 0
        return 0

    
    
    def fit(self,x,y=None):
        return self
    
    def transform(self,X):
        X_tagged=pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
def load_data(database_filepath):
    engine = create_engine('sqlite:///database_filename')
    df = pd.read_sql('SELECT * FROM InsertTableName3',engine)
    X = df.message.values
    y = df[df.columns[4:]]
    index = y[~y['related'].isnull()].index
    X=pd.DataFrame(X).iloc[index]
    y=y.iloc[index]
    return X, y, y.columns 

def tokenize(text):
 
    text=re.sub(r'[^a-zA-Z0-9]','',text) #remove punctuations
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower().strip())
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words and word != '']    
    return tokens



def build_model():
    
    pipeline = Pipeline([
        ('features', FeatureUnion([
            
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),
    
        ('clf',RandomForestClassifier())
    ])
    #parameters = {
        #'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        #'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        #'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        #'features__text_pipeline__tfidf__use_idf': (True, False),
        #'clf__n_estimators': [10, 20]
     #}
    #cv = GridSearchCV(pipeline, param_grid=parameters)
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred=model.predict(X_test)
    print (classification_report(Y_test,y_pred,target_names=category_names))


def save_model(model, model_filepath):
    from sklearn.externals import joblib
    c=joblib.dump(model,'''model_filepath''')


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X[0].values, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    
    main()