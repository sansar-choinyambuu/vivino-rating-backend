# vivino-rating-backend

REST API that provides access to machine learning model for wine rating prediction. Built with [flask](https://flask.palletsprojects.com/)

The machine learning model was trained with data scraped from [vivino.com](https://www.vivino.com/) using [vivino-scraper](https://github.com/sansar-choinyambuu/vivino-scraper)
More information in vivino data analysis in: https://github.com/sansar-choinyambuu/vivino-analysis 

The machine learning model is neural network classification model. The model is served via .h5 file in this repository. The input transformation is done via sklearn [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) which is also available via .pickle file in this repository. The transformers were especially useful to extract available categories for categorical input features.

## REST API endpoints
- GET /api/types - returns list of wine types i.e. red, white etc.
- GET /api/years - returns list of wine years i.e. 2018, 2012 etc.
- GET /api/grapes - returns list of main wine grapes i.e. malbec, nebbiolo  etc.
- GET /api/countries - returns list of countries i.e. argentina, italy etc.
- GET /api/regions - returns list of regions i.e. alsace-grand-cru, bordeaux etc.
- POST /api/rating - returns predicted wine rating. Expects input in following format:
```
{
	"price_chf": 20.0,
	"year": 2018,
	"type": "red",
	"country": "italy",
	"region": "tuscany",
	"main_grape": "primitivo",
	"cuvee": true
}
```