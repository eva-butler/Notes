# Intermediate Machine Learning

## Introduction
- tackle data types often found in real-world datasets (missing values, categorical variables),
- design pipelines to improve the quality of your machine learning code,
- use advanced techniques for model validation (cross-validation),
- build state-of-the-art models that are widely used to win Kaggle competitions (XGBoost), and
- avoid common and important data science mistakes (leakage).

here are the 5 models that they provide. Lets go through the difference of each one:
    
    # Define the models
    model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
    model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
    model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
    model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
    model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

n_estimators: refers to the number of trees in the forest. It controls how many decision trees
the model builds during training. Increasing n_estimators can improve model performance up to a point,
as it allows the model to better capture the patterns in the data. However, more trees also increase
computation time and can lead to diminishing returns. Generally, a higher number of trees leads to
more robust models, but you should balance it with computational resources.

random_state: controls the randomness of the estimator. It ensures that the results are reproducible
by setting a seed for the random number generator. This means that every time you run the model with
the same data and parameters, you'll get the same results. This is particularly useful for debugging
and sharing your results with others.

criterion='absolute_error': specifies that the model should use the mean absolute error (MAE) as the
criterion for splitting nodes. This means that the model will minimize the absolute differences between
the predicted and actual target values, making it more robust to outliers compared to criteria like mean
squared error (MSE).

min_samples_split=20: a node must have at least 20 samples before it can be split into further nodes.
This parameter helps control overfitting by preventing the model from creating splits that are too 
specific to the training data, ensuring that each split has a sufficient number of samples to be 
statistically meaningful.

max_depth=7: limits the maximum depth of each decision tree to seven levels. This parameter controls
the complexity of the model

score_model(): This function returns the mean absolute error (MAE) from the validation set.

from sklearn.metrics import mean_absolute_error

    # Function for comparing different models
    def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
        model.fit(X_t, y_t)
        preds = model.predict(X_v)
        return mean_absolute_error(y_v, preds)
    
    for i in range(0, len(models)):
        mae = score_model(models[i])
        print("Model %d MAE: %d" % (i+1, mae))


## Missing Values
