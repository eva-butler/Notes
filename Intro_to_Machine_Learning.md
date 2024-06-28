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
'''
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
'''
 
