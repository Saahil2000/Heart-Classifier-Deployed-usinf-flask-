import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../Heart.db')
df = pd.read_sql_table('MLStaging', engine)

# load model
model = joblib.load("../model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():

    # extract data needed for visuals
    #genre_counts = df.groupby('genre').count()['message']
    #genre_names = list(genre_counts.index)

    # extracting categories
    #category_names = df.iloc[:,5:].columns
    #category_boolean = (df.iloc[:,5:] != 0).sum().values

    # create visuals
    graphs = []
        # Graph1: Distribution of Message Genres

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query0 = request.args.get('query0', '')
    query1 = request.args.get('query1', '')
    query2 = request.args.get('query2', '')
    query3 = request.args.get('query3', '')
    query4 = request.args.get('query4', '')
    query5 = request.args.get('query5', '')
    query6 = request.args.get('query6', '')
    query7 = request.args.get('query7', '')
    query8 = request.args.get('query8', '')
    query9 = request.args.get('query9', '')
    query10 = request.args.get('query10', '')
    query11 = request.args.get('query11', '')
    query12 = request.args.get('query12', '')

    # use model to predict classification for query
    classification_labels = model.predict([[float(query0), float(query1), float(query2), float(query3), float(query4), float(query5), float(query6), float(query7), float(query8), float(query9), float(query10), float(query11), float(query12) ] ])
    # index of column start changed from 4 to 5 to accomadate my data storage

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        result = classification_labels[0]
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
