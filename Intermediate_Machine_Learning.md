# Intermediate Machine Learning
[course here](https://www.kaggle.com/learn/intermediate-machine-learning)
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
To test the different approaches for dealing with missing values, we will use this method to determine the quality of the approach:

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error
        
        # Function for comparing different approaches
        def score_dataset(X_train, X_valid, y_train, y_valid):
            model = RandomForestRegressor(n_estimators=100, random_state=0)
            model.fit(X_train, y_train)
            preds = model.predict(X_valid)
            return mean_absolute_error(y_valid, preds)
Three Approaches to Dealing with Missing Values:
1. Drop Columns

        #Get names of columns with missing values
        cols_with_missing = [col for col in X_train.columns
                             if X_train[col].isnull().any()]
        
        #Drop columns in training and validation data
        reduced_X_train = X_train.drop(cols_with_missing, axis=1)
        reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

2. Imputation:fills in the missing values with some number. For instance, we can fill in the mean value along each column.

from sklearn.impute import SimpleImputer

        #Imputation
        my_imputer = SimpleImputer()
        imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
        imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
        
        #Imputation removed column names; put them back
        imputed_X_train.columns = X_train.columns
        imputed_X_valid.columns = X_valid.columns
3. An Extension to Imputation: In this approach, we impute the missing values, as before. And, additionally, for each column with missing entries in the original dataset, we add a new column that shows the location of the imputed entries.

        # Make copy to avoid changing original data (when imputing)
        X_train_plus = X_train.copy()
        X_valid_plus = X_valid.copy()
        
        # Make new columns indicating what will be imputed
        for col in cols_with_missing:
            X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
            X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()
        
        # Imputation
        my_imputer = SimpleImputer()
        imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
        imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))
        
        # Imputation removed column names; put them back
        imputed_X_train_plus.columns = X_train_plus.columns
        imputed_X_valid_plus.columns = X_valid_plus.columns


## Categorical Variables

Its just a limited number of values. The number of values associated with that variable is considered its cardinality. There are three approaches to dealing with categorical variables
To get a list of the categorical variables:

        s = (X_train.dtypes == 'object')
        object_cols = list(s[s].index)
        
        print("Categorical variables:")
        print(object_cols)

 Define a function score_dataset():

     from sklearn.ensemble import RandomForestRegressor
     from sklearn.metrics import mean_absolute_error
    
    # Function for comparing different approaches
    def score_dataset(X_train, X_valid, y_train, y_valid):
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return mean_absolute_error(y_valid, preds)




    
1. Drop Categorical Variables

        drop_X_train = X_train.select_dtypes(exclude=['object'])
        drop_X_valid = X_valid.select_dtypes(exclude=['object'])
2. Ordinal Encoding: assigns each unique value to a different integer. Oridnal Variables sort of need to have a natural ordering to them that can be something like "Never" (0) < "Rarely" (1) < "Most days" (2) < "Every day" (3).

        from sklearn.preprocessing import OrdinalEncoder
        
        # Make copy to avoid changing original data 
        label_X_train = X_train.copy()
        label_X_valid = X_valid.copy()
        
        # Apply ordinal encoder to each column with categorical data
        ordinal_encoder = OrdinalEncoder()
        label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
        label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

3. One Hot Encoding: creates a new column indicating the presence or absense of a vairable. Variables that do not have a intrinsic ranking are considered nominal variables. We use the OneHotEncoder class from skikit-learn. There are many parameters that i need to look into later. Two of the ones mentionsed are handle_unknown (avoid errors when the valid data contains classess that are not represented in the training data) and sparse (ensures that the encoded columsn are returned as a numpy array [False]).

        from sklearn.preprocessing import OneHotEncoder
        
        # Apply one-hot encoder to each column with categorical data
        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
        OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))
        
        # One-hot encoding removed index; put it back
        OH_cols_train.index = X_train.index
        OH_cols_valid.index = X_valid.index
        
        # Remove categorical columns (will replace with one-hot encoding)
        num_X_train = X_train.drop(object_cols, axis=1)
        num_X_valid = X_valid.drop(object_cols, axis=1)
        
        # Add one-hot encoded columns to numerical features
        OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
        OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
        
        # Ensure all columns have string type
        OH_X_train.columns = OH_X_train.columns.astype(str)
        OH_X_valid.columns = OH_X_valid.columns.astype(str)


## Pipelines

Pipelines are a simple way to keep your data preprocessing and modeling code organized. Specifically, a pipeline bundles preprocessing and modeling steps so you can use the whole bundle as if it were a single step.

Step 1: Use ColumnTransformer class to bundle together different preprocessing steps. 
    
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    
    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant')
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
Step 2: define model

    from sklearn.ensemble import RandomForestRegressor
    
    model = RandomForestRegressor(n_estimators=100, random_state=0)

Step 3: definre a pipeline that bundles the preprocessing and modeling steps

With the pipeline, we preprocess the training data and fit the model in a single line of code. (In contrast, without a pipeline, we have to do imputation, one-hot encoding, and model training in separate steps. This becomes especially messy if we have to deal with both numerical and categorical variables!)
With the pipeline, we supply the unprocessed features in X_valid to the predict() command, and the pipeline automatically preprocesses the features before generating predictions. (However, without a pipeline, we have to remember to preprocess the validation data before making predictions.)
    
    from sklearn.metrics import mean_absolute_error
    
    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)
                                 ])
    
    # Preprocessing of training data, fit model 
    my_pipeline.fit(X_train, y_train)
    
    # Preprocessing of validation data, get predictions
    preds = my_pipeline.predict(X_valid)
    
    # Evaluate the model
    score = mean_absolute_error(y_valid, preds)
    print('MAE:', score)



