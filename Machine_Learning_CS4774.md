Table of Contents:
- Module 1: [Introduction to ML](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#module-1-introduction-to-ml)
    - [Class 1](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#class-1-in-person)
    - Readings: [Chapter 1](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#readings-chapter-1)
- Module 2: [Fundamentals of ML](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#module-2-fundamentals-of-ml)
    - [Class 2](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#class-2-in-person
  - Module 3: [End to end ML](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#module-3-end-to-end-ml)
    - Readings: [Chapter 2](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#readings-chapter-2)
    - [Class 3](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#class-3)
- Module 4: [Supervised Learning - Regression](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#module-4-supervised-learning---regression)
    - Reading: [Chapter 4]()
    - [Class 5](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#class-5-in-person)
    - [Video Notes]()
- Module 5/6: [Supervised Learning and Classification]()
  - Reading: [Chapter 3]()
  - [Lecture 5](): Classification
  - [Lecture 6](): Logisitic Regression
- Module 7: [Unsupervised Learning]()
  - Reading: [Chapter 9]()
  - [Lecture 7](): Clustering Methods HAC and k-means
- Module 8: [Decision Tree Learning]()
  - Reading: [Chapter 6]()
  - [Lecture 8](): Decision Trees
- Module 10: [Analogical Learning]()
  - Reading: [Chapter 5]()
  - [Lecture 10a]()
  - [Lecture 10b]()
- [MIDTERM REVIEW]()
 

# Module 1: Introduction to ML

## Class 1 (in person): 
- *Machine Learning*: The field of study that gives the computers the ability to learn with out being explicitly programmed

- Traditional Programming: Data and Program -> computer -> output
- Machine Learning: Data and Output -> computer -> program
- Machine Learning is a subfield of AI: AI > ML > Deep Learning
- Machine Learning Algorithm: to learn from experience with respect from some task T and some performance P if improves with E
  - The Task (T): learning is the means to achieve the task {regression, classification, transcription, machine translation, anomly detection, synthesis and sampling, density estimation}
  - The Performance (P): Quantitative measures to evaluate abilities of a Learning Algorithm {classifiction-> accuracy, regression->MSE, density-> log probabilities}
  - THe Experience (E): experience a dataset or examples divides into supervised and unsupervised learning, also reinforcement learning (supervised-> label {this is an apple}, unsupervised -> this looks like this other thing, reinforcement -> eat this and it keeps you from getting sick)

### Readings: Chapter 1
- Machine learning is the science (and art) of programming computers so they can learn from data.
- The examples that the system uses to learn are called the training set. Each training example is called a training instance (or sample). The part of a machine learning system that learns and makes predictions is called a model. 
- A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E. ****
- To summarize, machine learning is great for:
    - Problems for which existing solutions require a lot of fine-tuning or long lists of rules (a machine learning model can often simplify code and perform better than the traditional approach)
    - Complex problems for which using a traditional approach yields no good solution (the best machine learning techniques can perhaps find a solution)
    - Fluctuating environments (a machine learning system can easily be retrained on new data, always keeping it up to date)
    - Getting insights about complex problems and large amounts of data

- Supervised Learning: the training set you feed to the algorithm includes the desired solutions, called labels. A typical supervised learning task is classification. Another typical task is to predict a target numeric value, such as the price of a car, given a set of features
- Unsupervised Learning: In unsupervised learning, as you might guess, the training data is unlabeled The system tries to learn without a teacher. You may want to run a clustering algorithm to try to detect groups of similar visitors. If you use a hierarchical clustering algorithm, it may also subdivide each group into smaller groups. Visualization algorithms are also good examples of unsupervised learning: you feed them a lot of complex and unlabeled data, and they output a 2D or 3D representation of your data that can easily be plotted. A related task is dimensionality reduction, in which the goal is to simplify the data without losing too much information. For example, a car’s mileage may be strongly correlated with its age, so the dimensionality reduction algorithm will merge them into one feature that represents the car’s wear and tear. This is called *feature extraction*.  another important unsupervised task is anomaly detection—for example, detecting unusual credit card transactions to prevent fraud, catching manufacturing defects, or automatically removing outliers from a dataset before feeding it to another learning algorithm. A very similar task is novelty detection: it aims to detect new instances that look different from all instances in the training set. association rule learning, in which the goal is to dig into large amounts of data and discover interesting relations between attributes.
- Semi-supervised Learning: Since labeling data is usually time-consuming and costly, you will often have plenty of unlabeled instances, and few labeled instances. Some algorithms can deal with data that’s partially labeled. This is called semi-supervised learning.
- Self-supervised Learning: Another approach to machine learning involves actually generating a fully labeled dataset from a fully unlabeled one. Again, once the whole dataset is labeled, any supervised learning algorithm can be used. This approach is called self-supervised learning.
- Reinforcement Learning: Reinforcement learning is a very different beast. The learning system, called an agent in this context, can observe the environment, select and perform actions, and get rewards in return. It must then learn by itself what is the best strategy, called a policy, to get the most reward over time. A policy defines what action the agent should choose when it is in a given situation.

Batch Versus Online Learning:
- Batch Learning: In batch learning, the system is incapable of learning incrementally: it must be trained using all the available data. Normally done offline. first trained and then launched. performance decay over time.
- Online Learning: In online learning, you train the system incrementally by feeding it data instances sequentially, either individually or in small groups called mini-batches. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives.

Instance-Based Versus Model-Based Learning
- based on how a machine generalizes
- Instance-based learning: This is called instance-based learning: the system learns the examples by heart, then generalizes to new cases by using a similarity measure to compare them to the learned examples (or a subset of them).
- Model-based learning and a typical machine learning workflow: Another way to generalize from a set of examples is to build a model of these examples and then use that model to make predictions.

In summary:
- You studied the data.
- You selected a model.
- You trained it on the training data (i.e., the learning algorithm searched for the model parameter values that minimize a cost function).
- Finally, you applied the model to make predictions on new cases (this is called inference), hoping that this model will generalize well.

Main Challenges of Machine Learning:
- Insufficient Quantity of Training Data
- Nonrepresentative Training Data
- Poor-Quality Data
- Overfitting the Training Data
- Underfitting the Training data

Testing and Validating:
- training set and test set
- generalization error: error rate on new cases
- training error: errors on training set

Hyperparameter Tuning and Model Selection:
- train multiple models and compare them
- holdout validation: you simply hold out part of the training set to evaluate several candidate models and select the best one.
- validation set: the holdout set of data for determining which model performs the best (also called the dev set)

![mls3_0125](https://github.com/user-attachments/assets/efa05ce2-fab0-40a2-9e27-3a44b22ee572)

Data Mismatch:
- In this case, the most important rule to remember is that both the validation set and the test set must be as representative as possible of the data you expect to use in production, so they should be composed exclusively of representative pictures: you can shuffle them and put half in the validation set and half in the test set 
- One solution is to hold out some of the training pictures (from the web) in yet another set that Andrew Ng dubbed the train-dev set

No Free Lunch Theorem: A model is a simplified representation of the data. The simplifications are meant to discard the superfluous details that are unlikely to generalize to new instances. When you select a particular type of model, you are implicitly making assumptions about the data. 


# Module 2: Fundamentals of ML 

## Class 2 (in person):
- Traditional Approach: Study Problem -> Write Rules -> Evalutate -> Launch (Evaluate)
- Machine Learning Approach: Study Problem -> Train ML model -> evaluate -> analyze -> launch
  - Let the machine determine the rules
  - collect data over time and update over time; machine learning helps you understand the problem better -> data mining
 
- Types of Problems ML is good for:
    - hand tuning a list of rules
    - complex problems where there is no good traditional solution
    - changing env. that humans cant keep up with
    - large amounts of data

- CORRELATION NOT CAUSATION!!!!!!!!!!!
    - if your model does not make good predictions then you might just need to change the params of the model

- Challenges of ML:
    - DATA: Insufficient amount of data, natural language isambiguation, non-representative data, training data, have to be careful with outliers
        - sampling noise: error associated with sampling small dataset
        - sampling bias: large dataset is not representative due to a flawed sampling method
        - Poor quality data -> full of errors, missing data, outliers, noise ... cleaning data is very important
        - Irrelevant Features: GIGO principle -> must determine which features are relevant -> feature engineering, selection, extraction, creation
     - ALGORITHM: how to evaluate
         - Generalization: being able to perform well on unobserved input.
           - error measured on training set is a traininig error
           - Generalization error(test error): determining performance on the test set
           - Visualize a graph. AS capacity increases, training error goes down but generaralization error goes up. 
           - You want to minimize the gap between the training and test error
         - Model Capacity:ability to fit a wide variety of functions
             - low capacity: may struggle to fit training set (underfitting)
             - high capacity: overfit by memorizing everything in the training set and cant perform well on the unseen data (overfitting)
             - Overcome overfitting with regularization:
                 - controlled by hyperparameter (param of a LA, not model)
                 - hyperparameter must be set before training and remain constant
               - Testing and Validation:
                   - have to run different parameters to determine the best number of hyper params
                   - validate dev set
                   - 3 partitions of a set (training, validation, and test)
               - NO FREE LUNCH THEOREM (NFL) : a model is just a simplifed version of observerations of data. (ASSUMPTIONS) If you make no assumptions about data, then there is no reason to prefer one model over another.
               -  hypothesis space: the set of functions that the learning data algortihm us allowed to select as being the solution

# Module 3: End to End ML
### Readings: Chapter 2
This chapter is an example project end to end

[1](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#1-looking-at-the-big-picture)) Look at the big picture.

[2](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#2-get-the-data)) Get the data.

[3](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#3-explore-and-visualize-the-data-to-gain-insights)) Explore and visualize the data to gain insights.

[4](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#4-prepare-the-data-for-machine-learning-algorithms)) Prepare the data for machine learning algorithms.

[5](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#5-select-and-train-a-model)) Select a model and train it.

[6](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#6-fine-tune-your-model)) Fine-tune your model.

[7]()) Present your solution.

[8](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#8-present-launch-monitor-and-maintain-your-system)) Launch, monitor, and maintain your system.

 ####  (1) Looking at the Big Picture
Your first task is to use California census data to build a model of housing prices in the state.
- Frame the Problem:
    - Knowing the objective is important because it will determine how you frame the problem, which algorithms you will select, which performance measure you will use to evaluate your model, and how much effort you will spend tweaking it.
    - what the current solution looks like (if any). The current situation will often give you a reference for performance, as well as insights on how to solve the problem.
    - **pipelines**: A sequence of data processing components. Pipelines are very common in machine learning systems, since there is a lot of data to manipulate and many data transformations to apply.
 
  This is clearly a typical supervised learning task, since the model can be trained with labeled examples (each instance comes with the expected output, i.e., the district’s median housing price). It is a typical regression task, since the model will be asked to predict a value. More specifically, this is a multiple regression problem, since the system will use multiple features to make a prediction (the district’s population, the median income, etc.). It is also a univariate regression problem, since we are only trying to predict a single value for each district. If we were trying to predict multiple values per district, it would be a multivariate regression problem. Finally, there is no continuous flow of data coming into the system, there is no particular need to adjust to changing data rapidly, and the data is small enough to fit in memory, so plain batch learning should do just fine.
  
- Select a Performance Measure: A typical performance measure for regression problems is the root mean square error (RMSE). It gives an idea of how much error the system typically makes in its predictions, with a higher weight given to large errors.
  
![download](https://github.com/user-attachments/assets/18447aec-a1a7-4f28-8655-a17c0a8b27be)

- Although the RMSE is generally the preferred performance measure for regression tasks, in some contexts you may prefer to use another function. For example, if there are many outlier districts. In that case, you may consider using the mean absolute error (MAE, also called the average absolute deviation)

![images](https://github.com/user-attachments/assets/72162fc6-7fb3-4906-9be9-845a9c2f60b9)

- Check the Assumptions: Lastly, it is good practice to list and verify the assumptions that have been made so far (by you or others); this can help you catch serious issues early on.

#### (2) Get the Data

a function to fetch and load data:

        from pathlib import Path
        import pandas as pd
        import tarfile
        import urllib.request
        
        def load_housing_data():
            tarball_path = Path("datasets/housing.tgz")
            if not tarball_path.is_file():
                Path("datasets").mkdir(parents=True, exist_ok=True)
                url = "https://github.com/ageron/data/raw/main/housing.tgz"
                urllib.request.urlretrieve(url, tarball_path)
                with tarfile.open(tarball_path) as housing_tarball:
                    housing_tarball.extractall(path="datasets")
            return pd.read_csv(Path("datasets/housing/housing.csv"))
        
        housing = load_housing_data()

This give syou a histogram that shows the number of instances getting an idea of the distribution

        import matplotlib.pyplot as plt
        
        housing.hist(bins=50, figsize=(12, 8))
        plt.show()

- Finally, many histograms are skewed right: they extend much farther to the right of the median than to the left. This may make it a bit harder for some machine learning algorithms to detect patterns. Later, you’ll try transforming these attributes to have more symmetrical and bell-shaped distributions. > good to know. you want to try to have a normal distribution. i guess that makes it easier to spot patterns? Well we learned how to do that in Stat

Creating a Test Set:
- Creating a test set is theoretically simple; pick some instances randomly, typically 20% of the dataset (or less if your dataset is very large), and set them aside:

        import numpy as np
        
        def shuffle_and_split_data(data, test_ratio):
            shuffled_indices = np.random.permutation(len(data))
            test_set_size = int(len(data) * test_ratio)
            test_indices = shuffled_indices[:test_set_size]
            train_indices = shuffled_indices[test_set_size:]
            return data.iloc[train_indices], data.iloc[test_indices]

- need to ensure that the test set remains consistent over multiple runs. here is a wau to do that:

          from zlib import crc32
        
        def is_id_in_test_set(identifier, test_ratio):
            return crc32(np.int64(identifier)) < test_ratio * 2**32
        
        def split_data_with_id_hash(data, test_ratio, id_column):
            ids = data[id_column]
            in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
            return data.loc[~in_test_set], data.loc[in_test_set]

          housing_with_id = housing.reset_index()  # adds an `index` column
            train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")
  
- train_test_split() pretty much does the same thing as these methods above.

      from sklearn.model_selection import train_test_split
        train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

- want to put some strategy behind it to make sure the basic qualities of the sets are similar:

          strat_train_set, strat_test_set = train_test_split(
          housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

#### (3) Explore and Visualize the Data to Gain Insights
Only do this for the training set. do not touch the test set

Visualizing Geographical Data:
- good to create a scatter plot

        housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha = 0.2)
        plt.show()
- color mapping:

          housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
                     s=housing["population"] / 100, label="population",
                     c="median_house_value", cmap="jet", colorbar=True,
                     legend=True, sharex=False, figsize=(10, 7))
            plt.show()
Looking for Correlations:
- Since data set is kind of small you can compute the standard correlation coefficient between every pair of attributes using the corr() method. The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that there is a strong positive correlation; for example, the median house value tends to go up when the median income goes up. When the coefficient is close to –1, it means that there is a strong negative correlation; you can see a small negative correlation between the latitude and the median house value (i.e., prices have a slight tendency to go down when you go north). Finally, coefficients close to 0 mean that there is no linear correlation:
  
          corr_matrix = housing.corr()
- Another way is scatter_matrix(). plots every numerical attribute against every other numerical attribute. Since there are now 11 numerical attributes, you would get 112 = 121 plots, which would not fit on a page—so you decide to focus on a few promising attributes that seem most correlated with the median housing value

          from pandas.plotting import scatter_matrix
        
        attributes = ["median_house_value", "median_income", "total_rooms",
                      "housing_median_age"]
        scatter_matrix(housing[attributes], figsize=(12, 8))
        plt.show()

Experiment with Attribute Combinations
- you can just mess around wiht combos of attributes to see if you can make higher coorelations

#### (4) Prepare the Data for Machine Learning Algorithms
DO NOT DO THIS MANUALLY!! CREATE FUNCTIONS TO DO THIS FOR YOU!!
- This will allow you to reproduce these transformations easily on any dataset (e.g., the next time you get a fresh dataset).
- You will gradually build a library of transformation functions that you can reuse in future projects.
- You can use these functions in your live system to transform the new data before feeding it to your algorithms.
- This will make it possible for you to easily try various transformations and see which combination of transformations works best.

Clean the Data:
- Get rid of the corresponding districts.
- Get rid of the whole attribute.
- Set the missing values to some value (zero, the mean, the median, etc.). This is called imputation.
THese can be handled with dropna(), drop(), and fillna()
        
        housing.dropna(subset=["total_bedrooms"], inplace=True)  # option 1
        
        housing.drop("total_bedrooms", axis=1)  # option 2
        
        median = housing["total_bedrooms"].median()  # option 3
        housing["total_bedrooms"].fillna(median, inplace=True)

Imputation
- Simple Imputer
  
        from sklearn.impute import SimpleImputer
        
        imputer = SimpleImputer(strategy="median")
  - Missing values can also be replaced with the mean value (strategy="mean"), or with the most frequent value (strategy="most_frequent"), or with a constant value (strategy="constant", fill_value=…​). The last two strategies support non-numerical data.
- KNNImputer: replaces each missing value with the mean of the k-nearest neighbors’ values for that feature. The distance is based on all the available features.
- IterativeImputer: trains a regression model per feature to predict the missing values based on all the other available features. It then trains the model again on the updated data, and repeats the process several times, improving the models and the replacement values at each iteration.

Handling Text and Categorical Attributes
- OrdinalEncoder: meant for categorical variables

        from sklearn.preprocessing import OrdinalEncoder
        
        ordinal_encoder = OrdinalEncoder()
        housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
- OneHot Encoding: sort of picks on one variable. To fix this issue, a common solution is to create one binary attribute per category: one attribute equal to 1 when the category is "<1H OCEAN" (and 0 otherwise), another attribute equal to 1 when the category is "INLAND" (and 0 otherwise), and so on. This is called one-hot encoding, because only one attribute will be equal to 1 (hot), while the others will be 0 (cold). OneHotEncoder output one column per learned category, in the right order. 
        
        from sklearn.preprocessing import OneHotEncoder
        
        cat_encoder = OneHotEncoder()
        housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

Feature Scaling and Transformation: 
- min-max scaling:  is the simplest: for each attribute, the values are shifted and rescaled so that they end up ranging from 0 to 1. This is performed by subtracting the min value and dividing by the difference between the min and the max. Scikit-Learn provides a transformer called MinMaxScaler for this. It has a feature_range hyperparameter that lets you change the range if, for some reason, you don’t want 0–1

        from sklearn.preprocessing import MinMaxScaler
        
        min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
        housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
- Standardization: first it subtracts the mean value (so standardized values have a zero mean), then it divides the result by the standard deviation (so standardized values have a standard deviation equal to 1). Unlike min-max scaling, standardization does not restrict values to a specific range. This is like what we do in stats

        from sklearn.preprocessing import StandardScaler
        
        std_scaler = StandardScaler()
        housing_num_std_scaled = std_scaler.fit_transform(housing_num)

- TIP: When a feature’s distribution has a heavy tail (i.e., when values far from the mean are not exponentially rare), both min-max scaling and standardization will squash most values into a small range. Machine learning models generally don’t like this at all, as you will see in Chapter 4. So before you scale the feature, you should first transform it to shrink the heavy tail, and if possible to make the distribution roughly symmetrical. For example, a common way to do this for positive features with a heavy tail to the right is to replace the feature with its square root (or raise the feature to a power between 0 and 1). If the feature has a really long and heavy tail, such as a power law distribution, then replacing the feature with its logarithm may help.

Custom Transformers !! dont really understand this come back and read more thouroughly

you can create custom transformers like this. this is a log transformer:
        
        from sklearn.preprocessing import FunctionTransformer
        
        log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
        log_pop = log_transformer.transform(housing[["population"]])

Transformation Pipelines:
- the Pipelines class helps with the sequences of transformations. 

            from sklearn.pipeline import Pipeline
        
        num_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("standardize", StandardScaler()),
        ])
- The Pipeline constructor takes a list of name/estimator pairs (2-tuples) defining a sequence of steps. The names can be anything you like, as long as they are unique and don’t contain double underscores (__). They will be useful later, when we discuss hyperparameter tuning.
- If you don’t want to name the transformers, you can use the make_pipeline() function instead; it takes transformers as positional arguments and creates a Pipeline using the names of the transformers’ classes, in lowercase and without underscores (e.g., "simpleimputer"):

        from sklearn.pipeline import make_pipeline
        
        num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

- It would be more convenient to have a single transformer capable of handling all columns, applying the appropriate transformations to each column. For this, you can use a ColumnTransformer. For example, the following ColumnTransformer will apply num_pipeline (the one we just defined) to the numerical attributes and cat_pipeline to the categorical attribute:

        from sklearn.compose import ColumnTransformer
        
        num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
                       "total_bedrooms", "population", "households", "median_income"]
        cat_attribs = ["ocean_proximity"]
        
        cat_pipeline = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore"))
        
        preprocessing = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", cat_pipeline, cat_attribs),
        ])

  also

          from sklearn.compose import make_column_selector, make_column_transformer
        
        preprocessing = make_column_transformer(
            (num_pipeline, make_column_selector(dtype_include=np.number)),
            (cat_pipeline, make_column_selector(dtype_include=object)),
        )

        housing_prepared = preprocessing.fit_transform(housing)

- Now lets see what the full pipeline will look like:
    - Missing values in numerical features will be imputed by replacing them with the median, as most ML algorithms don’t expect missing values. In categorical features, missing values will be replaced by the most frequent category.
    - The categorical feature will be one-hot encoded, as most ML algorithms only accept numerical inputs.
    - A few ratio features will be computed and added: bedrooms_ratio, rooms_per_house, and people_per_house. Hopefully these will better correlate with the median house value, and thereby help the ML models.
    - A few cluster similarity features will also be added. These will likely be more useful to the model than latitude and longitude.
    - Features with a long tail will be replaced by their logarithm, as most models prefer features with roughly uniform or Gaussian distributions.
    - All numerical features will be standardized, as most ML algorithms prefer when all features have roughly the same scale.
            
            def column_ratio(X):
                return X[:, [0]] / X[:, [1]]
            
            def ratio_name(function_transformer, feature_names_in):
                return ["ratio"]  # feature names out
            
            def ratio_pipeline():
                return make_pipeline(
                    SimpleImputer(strategy="median"),
                    FunctionTransformer(column_ratio, feature_names_out=ratio_name),
                    StandardScaler())
            
            log_pipeline = make_pipeline(
                SimpleImputer(strategy="median"),
                FunctionTransformer(np.log, feature_names_out="one-to-one"),
                StandardScaler())
            cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
            default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                                 StandardScaler())
            preprocessing = ColumnTransformer([
                    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
                    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
                    ("people_per_house", ratio_pipeline(), ["population", "households"]),
                    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                                           "households", "median_income"]),
                    ("geo", cluster_simil, ["latitude", "longitude"]),
                    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
                ],
                remainder=default_num_pipeline)  # one column remaining:

#### (5) Select and Train a Model
Train and Evaluate on the Training Set:
- A linear regression model:

        from sklearn.linear_model import LinearRegression
        
        lin_reg = make_pipeline(preprocessing, LinearRegression())
        lin_reg.fit(housing, housing_labels)
- Decission Tree REgressor:

        from sklearn.tree import DecisionTreeRegressor
        
        tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
        tree_reg.fit(housing, housing_labels)

Better Evaluation Using Cross-Validation
- Scikit-Learn’s k_-fold cross-validation feature: The following code randomly splits the training set into 10 nonoverlapping subsets called folds, then it trains and evaluates the decision tree model 10 times, picking a different fold for evaluation every time and using the other 9 folds for training. The result is an array containing the 10 evaluation scores:

        from sklearn.model_selection import cross_val_score
        
        tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
                                      scoring="neg_root_mean_squared_error", cv=10)

- Random Forest Regressor:

        from sklearn.ensemble import RandomForestRegressor
        
        forest_reg = make_pipeline(preprocessing,
                                   RandomForestRegressor(random_state=42))
        forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                                        scoring="neg_root_mean_squared_error", cv=10)

#### (6) Fine Tune Your Model
Grid Search
- one option would be to mess with the hyperparams manually, but instead u can use Grid Search to do that for you
- GridSearchCV: tell it which hyperparameters you want it to experiment with and what values to try out, and it will use cross-validation to evaluate all the possible combinations of hyperparameter values

        from sklearn.model_selection import GridSearchCV
        
        full_pipeline = Pipeline([
            ("preprocessing", preprocessing),
            ("random_forest", RandomForestRegressor(random_state=42)),
        ])
        param_grid = [
            {'preprocessing__geo__n_clusters': [5, 8, 10],
             'random_forest__max_features': [4, 6, 8]},
            {'preprocessing__geo__n_clusters': [10, 15],
             'random_forest__max_features': [6, 8, 10]},
        ]
        grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                                   scoring='neg_root_mean_squared_error')
        grid_search.fit(housing, housing_labels)

- There are two dictionaries in this param_grid, so GridSearchCV will first evaluate all 3 × 3 = 9 combinations of n_clusters and max_features hyperparameter values specified in the first dict, then it will try all 2 × 3 = 6 combinations of hyperparameter values in the second dict. So in total the grid search will explore 9 + 6 = 15 combinations of hyperparameter values, and it will train the pipeline 3 times per combination, since we are using 3-fold cross validation. This means there will be a grand total of 15 × 3 = 45 rounds of training!

Randomized Search
- RandomizedSearchCV: often preferable when you have lots of differet possible combos.

        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint
        
        param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                          'random_forest__max_features': randint(low=2, high=20)}
        
        rnd_search = RandomizedSearchCV(
            full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
            scoring='neg_root_mean_squared_error', random_state=42)
        
        rnd_search.fit(housing, housing_labels)

Ensemble Models
- you can try to combine models that seem to work the best. The group (or “ensemble”) will often perform better than the best individual model—just like random forests perform better than the individual decision trees they rely on—especially if the individual models make very different types of errors.

Analyzing the Best models and their errors
- RandomForestRegressor -> good indication of the relative importance of each attribute for making accurate predictions
- You should also look at the specific errors that your system makes, then try to understand why it makes them and what could fix the problem: adding extra features or getting rid of uninformative ones, cleaning up outliers, etc.

Evalutate Your System on the Test Set
- run your final_model to transform the data and make predictions, then evaluate these predictions
        
        X_test = strat_test_set.drop("median_house_value", axis=1)
        y_test = strat_test_set["median_house_value"].copy()
        
        final_predictions = final_model.predict(X_test)
        
        final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
        print(final_rmse)  # prints 41424.40026462184

- you can compute a 95% confidence interval for the generalization error using scipy.stats.t.interval(). You get a fairly large interval from 39,275 to 43,467, and your previous point estimate of 41,424 is roughly in the middle of it:
        
        from scipy import stats
        >>> confidence = 0.95
        >>> squared_errors = (final_predictions - y_test) ** 2
        >>> np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
        ...                          loc=squared_errors.mean(),
        ...                          scale=stats.sem(squared_errors)))
        ...

#### (8) Present, Launch, Monitor, and Maintain Your System
- The most basic way to do this is just to save the best model you trained, transfer the file to your production environment, and load it.
        
        import joblib
        
        joblib.dump(final_model, "my_california_housing_model.pkl")

- Here is how you would use your model to make predictions:
        
        import joblib
        [...]  # import KMeans, BaseEstimator, TransformerMixin, rbf_kernel, etc.
        
        def column_ratio(X): [...]
        def ratio_name(function_transformer, feature_names_in): [...]
        class ClusterSimilarity(BaseEstimator, TransformerMixin): [...]
        
        final_model_reloaded = joblib.load("my_california_housing_model.pkl")
        
        new_data = [...]  # some new districts to make predictions for
        predictions = final_model_reloaded.predict(new_data)

- You need to put in some kind of monitoring system no mater how you launch the model


## Class 3
- **Looking at the Big Picture (1/8)**
  - frame the problem
    - task: regression, classification, clustering, visualization?
    - benefit???
    - performance? how to measure
        - Mean Absolute Error (MAE): measures distance between prediction and target values, corresponds to L1 Norm
               ![Screenshot 2024-09-04 102401](https://github.com/user-attachments/assets/6316e8b9-c995-4583-bc56-01ca188ce8f2)
              - X -> All the training data
              - h -> Hypothesis/model
              - m -> number of instances
              - x(i) -> feature vector of the ith instance
              - h(i) -> predicted value of the ith instance
              - y(i) -> value of the ith instance
        - Root Mean Squared Error (RMSE): Correspond to L2 norm, more sensitive to outliers than MSE, but generally performs better. -> seen errors appear LARGER than those in the MSE

          <img width="294" alt="image" src="https://github.com/user-attachments/assets/dc99ecad-003c-4848-a74a-69821a36fd34">
            - all are the same as they were for the MAE
 - ** Getting the Data (2/8)  **         
    - data: how much is available?
    - what learning algo should you use?
    - how mcuh effort should be spent?
    - what assumptions were mead
      - List and verify the assumptions that have been made so far for the model
      - Does the downstream system convert the prices to categories. in this case you might need to do this earlier rather than later. check with the team that is responsible for the downstream stuff

Mostly just running the colab file along with proff. 
- Use cut to change histograms with long tails into more bell shaped curves. THis is how he does it in the google collab:

        # Since this histogram is "tail heavy" (extends much farther to the right from the median and to the left)
        # We will try to transform this atrribute to have a more bell-shaped distribution
        housing["income_cat"] = pd.cut(housing["median_income"],
                                       bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                       labels=[1, 2, 3, 4, 5])

- stratified sampling -> ensures you keep the important rations.

            from sklearn.model_selection import StratifiedShuffleSplit
        
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]
- **Exploring and Visualizing the Data (3/8)**
  - in collab notebook
- **Cleaning the Data (4/8)**
  - takes the most of your time
  - Imputers cannot work with the categorical data. so you need to get rid of the columns that are not numbers and then add later
- what to do with the categorical data??
  - Ordinal Encoder -> just going to put a number for each category...some ML algorithms will assume two near by values are more similar than two distant vaues. This is misleading. In some cases this might actually be helpful. But for something like OCean proximity, the order of the categories does not map directly to the numberial categories
  - OneHotEncoder -> one attribute equal to 1 when the category is "< 1H OCEAN" (and 0 otherwise), another attribute equal to 1 when the category is “INLAND” (and 0 otherwise), and so on. This is called one-hot encoding, because only one attribute will be equal to 1 (hot), while the others will be 0 (cold).
  - Custom Transformer:Although Scikit-Learn provides many useful transformers, you will need to write your own for tasks such as custom cleanup operations or combining specific attributes. You will want your transformer to work seamlessly with Scikit-Learn functionalities (such as pipelines), and since Scikit-Learn relies on duck typing (not inheritance), all you need is to create a class and implement three methods: fit() (returning self), transform(), and fit_transform(). You can get the last one for free by simply adding TransformerMixin as a base class.
  - Standard Scalar: 

# Module 4: Supervised Learning - Regression 
### Readings: Chapter 4
Linear Regression:
- More generally, a linear model makes a prediction by simply computing a weighted
sum of the input features, plus a constant called the bias term (
- <img width="514" alt="image" src="https://github.com/user-attachments/assets/117cfeee-18bd-4cf1-97ec-0dae715d93d5">
- <img width="506" alt="image" src="https://github.com/user-attachments/assets/b6aaad1d-ba9b-4791-9bfc-a859b9093d1b">
- the most common performance measure
of a regression model is the Root Mean Square Error (RMSE) (Equation 2-1). There‐
fore, to train a Linear Regression model, you need to find the value of θ that minimi‐
zes the RMSE
- <img width="548" alt="image" src="https://github.com/user-attachments/assets/5fbb6ef4-318b-439c-8cfb-ab853c9a9243">

The Normal Equation:
- To find the value of θ that minimizes the cost function, there is a closed-form solution
—in other words, a mathematical equation that gives the result directly. This is called
the Normal Equation
<img width="344" alt="image" src="https://github.com/user-attachments/assets/0ffb490f-f5f4-40f1-95a5-ed5cac813dc0">


        X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

- Computational Complexity:
  - The Normal Equation computes the inverse of XT· X, which is an n × n matrix(where n is the number of features). The computational complexity of inverting such a matrix is typically about O(n2.4) to O(n3).

Gradient Descent:
- Gradient Descent is a very generic optimization algorithm capable of finding optimal
solutions to a wide range of problems. The general idea of Gradient Descent is to
tweak parameters iteratively in order to minimize a cost function
- An important parameter in Gradient Descent is the size of the steps, determined by
the learning rate hyperparameter. If the learning rate is too small, then the algorithm
will have to go through many iterations to converge, which will take a long time
- segment joining them never crosses the curve. This implies that there are no local
minima, just one global minimum. It is also a continuous function with a slope that
never changes abruptly.4
 These two facts have a great consequence: Gradient Descent
is guaranteed to approach arbitrarily close the global minimum (if you wait long
enough and if the learning rate is not too high).

Batch Gradient Desecent:
- you need to compute the gradient of the cost func‐
tion with regards to each model parameter θj. In other words, you need to calculatehow much the cost function will change if you change θj just a little bit. This is called a partial derivative.
- <img width="506" alt="image" src="https://github.com/user-attachments/assets/24cb0820-7eee-49ee-a483-a67705316120">
- <img width="489" alt="image" src="https://github.com/user-attachments/assets/885c6234-710f-4442-8157-dd86ee840ee7">
- <img width="437" alt="image" src="https://github.com/user-attachments/assets/17677661-4e15-4be1-b17d-f471ae0fccb6">

Stochastic Gradient Descent:
- Stochastic Gradient Descent has a better
chance of finding the global minimum than Batch Gradient Descent does.
- At the opposite extreme, Stochastic Gradient Descent just
picks a random instance in the training set at every step and computes the gradients
based only on that single instance.
- <img width="547" alt="image" src="https://github.com/user-attachments/assets/978a9ffd-3a47-43f2-b64e-b29b3528267e">

Mini-batch Gradient Descent:
- It is quite simple to understand once you know Batch and Stochastic Gradi‐
ent Descent: at each step, instead of computing the gradients based on the full train‐
ing set (as in Batch GD) or based on just one instance (as in Stochastic GD), MiniGradient Descent | 1218 While the Normal Equation can only perform Linear Regression, the Gradient Descent algorithms can beused to train many other models, as we will see.
batch GD computes the gradients on small random sets of instances called minibatches. The main advantage of Mini-batch GD over Stochastic GD is that you can
get a performance boost from hardware optimization of matrix operations, especially
when using GPUs.


Polynomial Regression:





## Class 5 (in person)
Data: From Table to Matrix
- m-> rows, n-> features

Linear Regression (Task)
<img width="466" alt="image" src="https://github.com/user-attachments/assets/f100cd89-a446-485f-acb3-f817ce06e792">
- x are the features and then they are associates with the model params theta
<img width="464" alt="image" src="https://github.com/user-attachments/assets/1ac51fbb-5055-41d7-a136-3306d60f3c46">
- it ends up being an equivalent form of a matrix multiplication
- theta is the model predictor. You just multiply the data set with the parameter and you end up with your predictions

<img width="457" alt="image" src="https://github.com/user-attachments/assets/537e2cb1-60b4-4669-8b7d-03a0342a9a34">
- You can implement this equation in one line of code
- inverting the matrix is n^3
- once you build though it runs fast

A Gradient-based Approach:
- solution to a wide range of problems
- calculate how much the loss function will change if we change the params a bit
<img width="395" alt="image" src="https://github.com/user-attachments/assets/751a163d-0ad0-457b-bd7f-67795650659c">
- i think its bsaically trying to determine the cost of going in each direction
- there is a minus sign becasue you always want to be moving to the point at which the minimizing error point is. So if you are in an area with positive slope you will be subtracting from theta which will help you descend to the point that minimizes the error. Same for the other side
- the learning rate determines your step size
<img width="455" alt="image" src="https://github.com/user-attachments/assets/773f62e6-12db-46a5-8eb5-70a7a296a964">
- demonstrates how important it is to determine the learning rate
- if MSE goes up with iterations, then your learning rate is too high
USE CODE FROM COLAB 3 for ASSIGNMENT 1

Stochastic Gradient Descent (SGD)
- instead of using the whole training set, SGD picks a **single random example** in the training set at every step and compute the gradients based on that example.
- It’s extremely fast, but is “stochastic” (random) in nature, its final parameter values are bounce around the minimum, which are good, but not optimal. Data point could be noisy which is no good

Mini-Batch gradient descent
- Instead of training on the full set (Batch GD) or based on just one sample (Scholastic GD), Mini-batch GD computes gradients on small random sets of samples (10-1000 in size) called mini-batches → best of both world

<img width="463" alt="image" src="https://github.com/user-attachments/assets/48adc699-438d-4f0c-a741-ec298c0bc28b">


# Module 5/6: Supervised Learning - Classification

## Class 6: Classification Problems
<img width="446" alt="image" src="https://github.com/user-attachments/assets/6907a651-ccff-498c-b1e5-e603704eb629">
<img width="466" alt="image" src="https://github.com/user-attachments/assets/97a8bd31-fecc-4523-85fe-d2761f110653">
- this is actually a really terrible classifier
- evaluating classifiers are a bit trickier than a regressor
- we will need to use things like the Confusion Matrix, Precision and Recall, F-1 Score, ROC curve, and Area under the ROC

**just watch the videos for this section**




