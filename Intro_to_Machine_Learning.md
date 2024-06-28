# Intro to Machine Learning
- Kaggle Course: [here](https://www.kaggle.com/learn/intro-to-machine-learning)

Decision Tree: 
- the step of capturing patterns is called fitting or training a model
- the data used to fit the model is the training data
- you then use the model to predict the future outcomes of events

Any machine learning project begins with understanding the data you are working with.
Interpreting the data description using the .describe() method
- count: shows how many rows have non missing values
- mean: avg
- std: standard deviation
- ect.

Selecting Data For Modeling:
- features: variables that are inputted into the model that are later used to make predictions


melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
y = melbourne_data.Price

y-> is the target. what you are trying to predict

Building your Model:
- scikit-learn library is one of the most popular machine learning libraries

- 
"Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
Fit: Capture patterns from provided data. This is the heart of modeling.
Predict: Just what it sounds like
Evaluate: Determine how accurate the model's predictions are."

from sklearn.tree import DecisionTreeRegressor
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)


How you use the model:

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))


Model Validation:
- The relevant measure of model quality is predictive accuracy. Will the model's predictions be close to what actually happens
- MAE (mean abs error): take the abs value of each error and then take the avg of those abs errors. this is considered an in sample score. we used a the same sample to both build and evaluate the model.
- error=actual−predicted


how to calculate MAE: 
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

- an alternative to in sample validation, you can use validation data, which is basically reserved in the beginning and not used to train the model.

- train_test_split: is used to break up the data into two pieces. 
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
#Define model
melbourne_model = DecisionTreeRegressor()
#Fit model
melbourne_model.fit(train_X, train_y)

#get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


Underfitting and Overfitting:
- overfitting: where a model matches the training data almost perfectly, but does a poor job with new data
- underfitting: when a model fails to capture important distinctions and patterns in the data

- controlling tree depth: uses max_leaf_nodes as an argument
