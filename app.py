from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import session

import pickle
import pandas as pd
import numpy as np

from keras.models import load_model

MODEL_FILE = "vivino_rating_classifier_keras_model.h5"
PIPELINE_FILE = "transformation_pipeline.pickle"

RATINGS = {0: "average", 1: "good", 2: "very good", 3: "extra ordinary"}

app = Flask(__name__)
CORS(app)

classifier = load_model(MODEL_FILE)
pipeline = pickle.load(open(PIPELINE_FILE, "rb"))

def get_categories_from_pipeline(transformer_name):
    categories = pipeline.named_transformers_[transformer_name].categories[0]
    categories = list(categories) # to convert from pandas categorical series
    return jsonify(categories)

@app.route('/api/types', methods=['GET'])
def get_types():
    return get_categories_from_pipeline("type_encoding")

@app.route('/api/years', methods=['GET'])
def get_years():
    return get_categories_from_pipeline("year_encoding")

@app.route('/api/grapes', methods=['GET'])
def get_grapes():
    return get_categories_from_pipeline("main_grape_encoding")

@app.route('/api/countries', methods=['GET'])
def get_countries():
    return get_categories_from_pipeline("country_encoding")

@app.route('/api/regions', methods=['GET'])
def get_regions():
    return get_categories_from_pipeline("region_encoding")


@app.route('/api/rating', methods=['POST'])
def predict_wine_rating():

    # transform the inputs through the same pipeline as model training data
    input_df = pd.DataFrame(request.get_json(), index = [0])
    input_df["year"] = input_df["year"].astype("object")
    transformed = pipeline.fit_transform(input_df)

    # make prediction using loaded model
    prediction = classifier.predict(transformed)

    # convert to human readable rating
    rating = RATINGS[np.argmax(prediction)]
    return jsonify({'rating': rating})

if __name__ == "__main__":
    app.run()