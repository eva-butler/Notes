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
    - Reading: [Chapter 4](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#readings-chapter-4)
    - [Class 5](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#class-5-in-person)
    - [Video Notes](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#video-notes)
- Module 5/6: [Supervised Learning and Classification](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#module-56-supervised-learning---classification)
  - Reading: [Chapter 3](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#reading-chapter-3)
  - [Lecture 5 Videos](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#lecture-5-videos): Classification
  - [Lecture 6 Videos](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#lecture-6-videos): Logisitic Regression
- Module 7: [Unsupervised Learning](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#module-7-unsupervised-learning)
  - Reading: [Chapter 9]()
  - [Lecture 7](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#lecture-7-videos-clustering-methods-hac-and-k-means): Clustering Methods HAC and k-means
- Module 8: [Decision Tree Learning](https://github.com/eva-butler/Notes/blob/main/Machine_Learning_CS4774.md#module-8-decision-tree-learning)
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
- you can use linear model to fit nonlinear dat
- A simple way to do this is to
add powers of each feature as new features, then train a linear model on this extended
set of features. This technique is called Polynomial Regression.
- <img width="506" alt="image" src="https://github.com/user-attachments/assets/280ff306-314c-4257-8e6d-8947db0cc2ce">

Learning Curves:
- Another way is to look at the learning curves: these are plots of the model’s perfor‐
mance on the training set and the validation set as a function of the training set size
(or the training iteration). To generate the plots, simply train the model several times
on different sized subsets of the training set. The following code defines a function
that plots the learning curves of a model given some training data:
- <img width="486" alt="image" src="https://github.com/user-attachments/assets/8cac3aa9-6425-444e-8944-6500ab55d3ed">
- Underfitting model:
- <img width="554" alt="image" src="https://github.com/user-attachments/assets/34be580e-10d8-46c4-8585-ebed920607dc">
- Overfitting Model:
- <img width="536" alt="image" src="https://github.com/user-attachments/assets/7f7c2dc6-2e84-4e04-ad9a-c2322f3e1c05">


Regularized Linear Models:
- reduces over fitting
- the fewer degrees of freedom, the harder it is to overfit the data

Ridge Regression:
Ridge Regression (also called Tikhonov regularization) is a regularized version of Lin‐
ear Regression: a regularization term equal to α∑n to i = 1 | θi^2 is added to the cost function.
This forces the learning algorithm to not only fit the data but also keep the model
weights as small as possible. Note that the regularization term should only be added
to the cost function during training. Once the model is trained, you want to evaluate
the model’s performance using the unregularized performance measure
<img width="335" alt="image" src="https://github.com/user-attachments/assets/52f70617-6e13-4ecc-a9a1-a5f8a68d15c9">
<img width="557" alt="image" src="https://github.com/user-attachments/assets/f21a6200-c6a4-420c-928d-32781939f080">

Lasso Regression:
- Least Absolute Shrinkage and Selection Operator Regression (simply called Lasso
Regression) is another regularized version of Linear Regression: just like Ridge
Regression, it adds a regularization term to the cost function, but it uses the ℓ1 norm
of the weight vector instead of half the square of the ℓ2 norm
<img width="314" alt="image" src="https://github.com/user-attachments/assets/346eeb29-fef6-47df-85ce-ec8aba6d2a55">
<img width="532" alt="image" src="https://github.com/user-attachments/assets/1dd4c8f5-3a85-4b31-80a2-275962c3375f">

Elastic Net:
- Elastic Net is a middle ground between Ridge Regression and Lasso Regression. The
regularization term is a simple mix of both Ridge and Lasso’s regularization terms,
and you can control the mix ratio r. When r = 0, Elastic Net is equivalent to Ridge
Regression, and when r = 1, it is equivalent to Lasso Regression
<img width="365" alt="image" src="https://github.com/user-attachments/assets/23e6d1fa-d3ba-4f58-a03c-cb3f44a1dff1">
<img width="528" alt="image" src="https://github.com/user-attachments/assets/01f8f99b-bbc4-49c1-9ba4-7f7362c08536">


- So when should you use plain Linear Regression (i.e., without any regularization),
Ridge, Lasso, or Elastic Net? It is almost always preferable to have at least a little bit of
regularization, so generally you should avoid plain Linear Regression. Ridge is a good
default, but if you suspect that only a few features are actually useful, you should pre‐
fer Lasso or Elastic Net since they tend to reduce the useless features’ weights down to
zero as we have discussed. In general, Elastic Net is preferred over Lasso since Lasso
may behave erratically when the number of features is greater than the number of
training instances or when several features are strongly correlated.

Early Stopping
- A very different way to regularize iterative learning algorithms such as Gradient
Descent is to stop training as soon as the validation error reaches a minimum. This is
called early stopping
<img width="536" alt="image" src="https://github.com/user-attachments/assets/ef5ad53a-9d97-4776-9a8a-54adef493d54">
<img width="540" alt="image" src="https://github.com/user-attachments/assets/da2dcdb0-2df2-439f-a58b-6bf41fc08f20">

Logistic Regression:
-  Logistic Regression (also called Logit Regression) is com‐
monly used to estimate the probability that an instance belongs to a particular class
(e.g., what is the probability that this email is spam?). If the estimated probability is
greater than 50%, then the model predicts that the instance belongs to that class
(called the positive class, labeled “1”), or else it predicts that it does not (i.e., it
belongs to the negative class, labeled “0”). This makes it a binary classifier.

Estimating Probabilities:
- a Logistic Regression
model computes a weighted sum of the input features (plus a bias term), but instead
of outputting the result directly like the Linear Regression model does, it outputs the
logistic of this result
<img width="504" alt="image" src="https://github.com/user-attachments/assets/2ef1258b-a4b5-498d-a8bc-162992ade4b3">
<img width="296" alt="image" src="https://github.com/user-attachments/assets/a9bb69c2-6b8a-45a1-9c81-2e8cc1643379">
- The logistic—also called the logit, noted σ(·)—is a sigmoid function (i.e., S-shaped)
that outputs a number between 0 and 1. 
<img width="382" alt="image" src="https://github.com/user-attachments/assets/b00de3a2-3282-4b5a-9a59-9f880cef3732">

Training and Cost function:
- The objective of training is to set the param‐
eter vector θ so that the model estimates high probabilities for positive instances (y = 1) and low probabilities for negative instances (y = 0).
- <img width="437" alt="image" src="https://github.com/user-attachments/assets/5d739574-6b03-4820-aa0f-927b48155e59">
- This cost function makes sense because – log(t) grows very large when t approaches
0, so the cost will be large if the model estimates a probability close to 0 for a positive
instance, and it will also be very large if the model estimates a probability close to 1
for a negative instance. On the other hand, – log(t) is close to 0 when t is close to 1, so
the cost will be close to 0 if the estimated probability is close to 0 for a negative
instance or close to 1 for a positive instance, which is precisely what we want.

- The cost function over the whole training set is simply the average cost over all train‐
ing instances
<img width="361" alt="image" src="https://github.com/user-attachments/assets/fe392b3e-ed6d-4743-8edd-b76e310dc1cd">
<img width="374" alt="image" src="https://github.com/user-attachments/assets/a0d18105-eea3-4460-97aa-1bc61a753123">
- using gradient descent to determine what minimizies the cost function

Decision Boundaries:
- <img width="533" alt="image" src="https://github.com/user-attachments/assets/a2fe1277-c3e0-4c52-892a-22bf653723d5">
- just to get an idea of what a decision boundary looks like

Softmax Regression:
- The Logistic Regression model can be generalized to support multiple classes directly,
without having to train and combine multiple binary classifiers (as discussed in
Chapter 3). This is called Somax Regression, or Multinomial Logistic Regression
- The idea is quite simple: when given an instance x, the Softmax Regression model
first computes a score sk(x) for each class k, then estimates the probability of each
class by applying the somax function (also called the normalized exponential) to the
scores.
<img width="314" alt="image" src="https://github.com/user-attachments/assets/2d00c52e-ea18-4527-b149-2f8cdd9a9c0e">
<img width="233" alt="image" src="https://github.com/user-attachments/assets/e21f5f21-ca25-4587-99bb-750d3a92fdd1">
• K is the number of classes.
• s(x) is a vector containing the scores of each class for the instance x.
• σ(s(x))k
 is the estimated probability that t
<img width="388" alt="image" src="https://github.com/user-attachments/assets/a1f31df7-b387-4cba-a4a9-99f8afc835e2">
- Now that you know how the model estimates probabilities and makes predictions,
let’s take a look at training. The objective is to have a model that estimates a high
probability for the target class (and consequently a low probability for the other
classes). Minimizing the cost function shown in Equation 4-22, called the cross
entropy, should lead to this objective because it penalizes the model when it estimates a low probability for a target class. Cross entropy is frequently used to measure how
well a set of estimated class probabilities match the target classes
- <img width="515" alt="image" src="https://github.com/user-attachments/assets/d097cefb-5f40-4032-97e9-844c498a46f9">
- <img width="515" alt="image" src="https://github.com/user-attachments/assets/7e1b3580-f99d-4ea4-8a1d-18598343ff5f">
- Now you can compute the gradient vector for every class, then use Gradient Descent
(or any other optimization algorithm) to find the parameter matrix Θ that minimizes
the cost function
- Let’s use Softmax Regression to classify the iris flowers into all three classes. ScikitLearn’s LogisticRegression uses one-versus-all by default when you train it on more
than two classes, but you can set the multi_class hyperparameter to "multinomial"
to switch it to Softmax Regression instead. You must also specify a solver that sup‐
ports Softmax Regression, such as the "lbfgs" solver (see Scikit-Learn’s documenta‐
tion for more details). It also applies ℓ2 regularization by default, which you can control using the hyperparameter C.
<img width="526" alt="image" src="https://github.com/user-attachments/assets/5a9882bd-6e04-4096-9d27-25f6c7a5ca8a">
- Decision oundaries:
<img width="536" alt="image" src="https://github.com/user-attachments/assets/61f389f3-8a2b-42ea-bb6b-490ceb19678c">


## Video Notes:

### Video 4.1: Optimizing Model Parameters
- You can transform a table into a matrix so that we can do math on it
- a single row is a data point. Can consist of many different features that are for a single example.
- We also have the label for the example labeled y
- we have m examples and n features.
- We use subscripts to denote the identity of the features.
<img width="305" alt="image" src="https://github.com/user-attachments/assets/ddcf17e9-307c-42b7-87d2-5a1889ccc92e">

Task: Linear Regression:
- we have the generalized linear model:
<img width="306" alt="image" src="https://github.com/user-attachments/assets/72a254d4-321f-4e62-97f9-2cbe1aa7f7d7">
- x0 is always equal to one because theta 0 is going to be the biased term.
- we can also write it in the vector form:
<img width="288" alt="image" src="https://github.com/user-attachments/assets/51f368ef-476b-4464-9298-cdbb2b5b1814">
Representation: Linear function: h(theta) : X -> y
<img width="314" alt="image" src="https://github.com/user-attachments/assets/2272701c-28f4-46eb-b5d4-a54d1fe18812">
- we need to learn theta. in order to do this we are going to learn theta using a loss function and we want theta to be the point at which it is minimized.

Loss Function-> MSE
<img width="293" alt="image" src="https://github.com/user-attachments/assets/8fcbbaf9-fc63-41b8-99f8-53ab389e0c1d">

minimizing the function:
<img width="300" alt="image" src="https://github.com/user-attachments/assets/e5045359-6785-47cb-877e-8dd676c3512b">
- we can go a little further:
<img width="308" alt="image" src="https://github.com/user-attachments/assets/51b0c058-8d6d-4e21-807d-595958585abe">
Optimization Procedure:
- we now want to minimize the function J. To do this you take the partial derivative with respect to the parameter of the function and then setting it equal to 0
<img width="292" alt="image" src="https://github.com/user-attachments/assets/3e2e50e0-d6f7-4ca2-b626-210624577110">

NORMAL EQUATION:
<img width="299" alt="image" src="https://github.com/user-attachments/assets/7e5af061-43a2-4a5d-949f-ed0cbf3b48f0">

Computational Complexity:
<img width="305" alt="image" src="https://github.com/user-attachments/assets/f2dd9c6c-b694-47fd-a5db-3e419948a4c3">
<img width="296" alt="image" src="https://github.com/user-attachments/assets/19aa00c9-34b6-4ef7-9d99-5cb7377722c3">

### Video 4.2: Alternative Optimization: Gradient Based Approach
- the gradient will tell you how far to go in the direction of fastest descent

Gradient Descent (GD): 
- an optimization alg. to find solution
<img width="311" alt="image" src="https://github.com/user-attachments/assets/e33f9f4d-3804-4d10-836a-c4b83b7bcfa9">

Batch Gradient Descent (BGD):
- tweak the parameters iteratively in order to minimize the loss functino
- You use ALL of the training data. (BATCH)
- determined by the learning rate
<img width="271" alt="image" src="https://github.com/user-attachments/assets/2f39fa57-6156-4585-85fd-c762782247a9">
- Gradient Descent works better when your features are all scaled. It makes it work much faster and the descent is much faster.
- FORMULAS:
<img width="292" alt="image" src="https://github.com/user-attachments/assets/c2ca748d-8ba4-4f78-a509-4f15ff6c3f1e">
^this is all for a single direction theta j
- now we want to generalize and look at all directions so we can calculate the graident vector:
<img width="301" alt="image" src="https://github.com/user-attachments/assets/86e81dba-6ae5-413f-a938-b7981b4c3553">
Gradient Descent Step:
<img width="277" alt="image" src="https://github.com/user-attachments/assets/14af58ec-c59f-4044-b475-adb67323108c">

Stochastic Gradient Descent (SGD):
- instead of using the entire training set, we just pick a random group of the training data and computes the gradients at each step.
<img width="314" alt="image" src="https://github.com/user-attachments/assets/55002742-c7df-4a2e-a2cb-22c340a7bfda">

Mini-bath Gradient Descent:
- sort of a compromise between the two. speeds up the training process and not quiet as random:
<img width="299" alt="image" src="https://github.com/user-attachments/assets/fad56119-32ad-4368-abe7-3470c2f47cc1">

<img width="314" alt="image" src="https://github.com/user-attachments/assets/e62fc4d5-1b96-40d2-a505-fe1de96b5c3c">
- Noraml is our closed form 



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

## Reading: Chapter 3

Training a Binary Classifier:
- This “5-detector” will be an example of a binary classifier, capable of
distinguishing between just two classes, 5 and not-5. Let’s create the target vectors for
this classification task:

        y_train_5 = (y_train == 5) # True for all 5s, False for all other digits.
        y_test_5 = (y_test == 5)
- A good place to start is with a StochasticGradient Descent (SGD) classifier, using Scikit-Learn’s SGDClassifier class. This clas‐sifier has the advantage of being capable of handling very large datasets efficiently.This is in part because SGD deals with training instances independently, one at a time (which also makes SGD well suited for online learning), as we will see later.

        from sklearn.linear_model import SGDClassifier
        sgd_clf = SGDClassifier(random_state=42)
        sgd_clf.fit(X_train, y_train_5)
        >>> sgd_clf.predict([some_digit])
        array([ True], dtype=bool)

Performance Measures:

- Cross-Validation:
  - Let’s use the cross_val_score() function to evaluate your SGDClassifier model
using K-fold cross-validation, with three folds. Remember that K-fold crossvalidation means splitting the training set into K-folds (in this case, three), then mak‐
ing predictions and evaluating them on each fold using a model trained on the
remaining folds
  - <img width="458" alt="image" src="https://github.com/user-attachments/assets/2e00efea-971f-4b7c-ae76-5640126bca50">
  - <img width="516" alt="image" src="https://github.com/user-attachments/assets/28efea08-82ff-4c8e-9839-0c875e2d6d7c">

- Confusion Matrix
  - A much better way to evaluate the performance of a classifier is to look at the confu‐
sion matrix. The general idea is to count the number of times instances of class A are
classified as class B. For example, to know the number of times the classifier confused
images of 5s with 3s, you would look in the 5th row and 3rd column of the confusion
matrix.
  - To compute the confusion matrix, you first need to have a set of predictions, so they
can be compared to the actual targets. You could make predictions on the test set, but
let’s keep it untouched for now (remember that you want to use the test set only at the
very end of your project, once you have a classifier that you are ready to launch).

        from sklearn.model_selection import cross_val_predict
        y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
  - Just like the cross_val_score() function, cross_val_predict() performs K-fold
cross-validation, but instead of returning the evaluation scores, it returns the predic‐
tions made on each test fold. This means that you get a clean prediction for each
instance in the training set (“clean” meaning that the prediction is made by a model
that never saw the data during training)
  - Now you are ready to get the confusion matrix using the confusion_matrix() func‐
tion. Just pass it the target classes (y_train_5) and the predicted classes
(y_train_pred):

        >>> from sklearn.metrics import confusion_matrix
        >>> confusion_matrix(y_train_5, y_train_pred)
        array([[53272, 1307],
         [ 1077, 4344]])

- The Preciscion of the Classifier:
  - <img width="170" alt="image" src="https://github.com/user-attachments/assets/bc5d0e24-dcdf-4285-ad0d-649da662ef03">
  - TP is the number of true positives, and FP is the number of false positives.
  - An interesting one to look at is the accuracy of the positive pre‐
dictions


- The Recall of a Classifier:
  - <img width="161" alt="image" src="https://github.com/user-attachments/assets/b9442200-141f-40eb-8e3e-4eb1ba61df52">
  - FN is of course the number of false negatives.
  -  also called sensitivity or true positive rate (TPR): this is the ratio of positive instances that are correctly detected by the classifier

 - PRECISION AND RECALL:
<img width="511" alt="image" src="https://github.com/user-attachments/assets/f1f1333d-a98e-4cc5-afa4-bdc9a75104c2">

        >>> from sklearn.metrics import precision_score, recall_score
        >>> precision_score(y_train_5, y_train_pred) # == 4344 / (4344 + 1307)
        0.76871350203503808
        >>> recall_score(y_train_5, y_train_pred) # == 4344 / (4344 + 1077)
        0.80132816823464303

- The F1 Score:
  - The combination of Precision and Recall
  - if you need a simple way to compare two classifiers.
  -  The F1 score isthe harmonic mean of precision and recall (Equation 3-3). Whereas the regular mean treats all values equally, the harmonic mean gives much more weight to low values. As a result, the classifier will only get a high F1 score if both recall and precision are high.
  -  <img width="409" alt="image" src="https://github.com/user-attachments/assets/f6989e5e-42a3-485d-b150-54400fb81437">

            >>> from sklearn.metrics import f1_score
            >>> f1_score(y_train_5, y_train_pred)
            0.78468208092485547
- The F1 score favors classifiers that have similar precision and recall. This is not always what you want: in some contexts you mostly care about precision, and in other contexts you really care about recall

- Precision/Recall Tradeoff:
  - you can’t have it both ways: increasing precision reduces recall, and vice versa. This is called the precision/recall tradeo.
  - Scikit-Learn does not let you set the threshold directly, but it does give you access to
the decision scores that it uses to make predictions. Instead of calling the classifier’s
predict() method, you can call its decision_function() method, which returns a
score for each instance, and then make predictions based on those scores using any
threshold you want:

        >>> y_scores = sgd_clf.decision_function([some_digit])
        >>> y_scores
        array([ 161855.74572176])
        >>> threshold = 0
        >>> y_some_digit_pred = (y_scores > threshold)
        array([ True], dtype=bool)
  - The SGDClassifier uses a threshold equal to 0, so the previous code returns the same
result as the predict() method (i.e., True). Let’s raise the threshold:

        >>> threshold = 200000
        >>> y_some_digit_pred = (y_scores > threshold)
        >>> y_some_digit_pred
        array([False], dtype=bool)
  - This confirms that raising the threshold decreases recall. The image actually repre‐
sents a 5, and the classifier detects it when the threshold is 0, but it misses it when the
threshold is increased to 200,000.

  - For this you will first need to get the
scores of all instances in the training set using the cross_val_predict() function
again, but this time specifying that you want it to return decision scores instead of
predictions:

        y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
         method="decision_function")
        
        from sklearn.metrics import precision_recall_curve
        precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
        
        def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
         plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
         plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
         plt.xlabel("Threshold")
         plt.legend(loc="center left")
         plt.ylim([0, 1])
        plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
        plt.show()
 - <img width="533" alt="image" src="https://github.com/user-attachments/assets/29d2627a-955a-44bd-9ebf-c85396e3f731">
 - Now you can simply select the threshold value that gives you the best precision/recall
tradeoff for your task. Another way to select a good precision/recall tradeoff is to plot
precision directly against recall,
   - <img width="498" alt="image" src="https://github.com/user-attachments/assets/5c9c7686-846a-40ed-b0b9-8b7b8df5237d">
   
- The ROC Curve:
  - The receiver operating characteristic (ROC) curve is another common tool used with
binary classifiers. It is very similar to the precision/recall curve, but instead of plot‐
ting precision versus recall, the ROC curve plots the true positive rate (another name
for recall) against the false positive rate.The FPR is the ratio of negative instances that
are incorrectly classified as positive. It is equal to one minus the true negative rate,
which is the ratio of negative instances that are correctly classified as negative. The
TNR is also called specificity. Hence the ROC curve plots sensitivity (recall) versus
1 – specificity.
  - To plot the ROC curve, you first need to compute the TPR and FPR for various thres‐
hold values, using the roc_curve() function:

        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
        def plot_roc_curve(fpr, tpr, label=None):
         plt.plot(fpr, tpr, linewidth=2, label=label)
         plt.plot([0, 1], [0, 1], 'k--')
         plt.axis([0, 1, 0, 1])
         plt.xlabel('False Positive Rate')
         plt.ylabel('True Positive Rate')
        plot_roc_curve(fpr, tpr)
        plt.show()

  - Once again there is a tradeoff: the higher the recall (TPR), the more false positives
(FPR) the classifier produces. The dotted line represents the ROC curve of a purely
random classifier; a good classifier stays as far away from that line as possible (toward
the top-left corner).
  -  area under the curve (AUC). A per‐
fect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will
have a ROC AUC equal to 0.5. Scikit-Learn provides a function to compute the ROC
AUC:
            
            >>> from sklearn.metrics import roc_auc_score
            >>> roc_auc_score(y_train_5, y_scores)
            0.96244965559671547


Multiclass Classification:
- Whereas binary classifiers distinguish between two classes, multiclass classifiers (also
called multinomial classifiers) can distinguish between more than two classes.
- For example, one way to create a system that can classify the digit images into 10
classes (from 0 to 9) is to train 10 binary classifiers, one for each digit (a 0-detector, a
1-detector, a 2-detector, and so on). Then when you want to classify an image, you get
the decision score from each classifier for that image and you select the class whose
classifier outputs the highest score. This is called the one-versus-all (OvA) strategy
(also called one-versus-the-rest).
- Another strategy is to train a binary classifier for every pair of digits: one to distin‐
guish 0s and 1s, another to distinguish 0s and 2s, another for 1s and 2s, and so on.
This is called the one-versus-one (OvO) strategy. If there are N classes, you need to
train N × (N – 1) / 2 classifiers. For the MNIST problem, this means training 45
binary classifiers! When you want to classify an image, you have to run the image
through all 45 classifiers and see which class wins the most duels. The main advan‐
tage of OvO is that each classifier only needs to be trained on the part of the training
set for the two classes that it must distinguish.
- Scikit-Learn detects when you try to use a binary classification algorithm for a multi‐
class classification task, and it automatically runs OvA (except for SVM classifiers for
which it uses OvO). Let’s try this with the SGDClassifier:

            >>> sgd_clf.fit(X_train, y_train) # y_train, not y_train_5
            >>> sgd_clf.predict([some_digit])
            array([ 5.])
- If you want to force ScikitLearn to use one-versus-one or one-versus-all, you can use
the OneVsOneClassifier or OneVsRestClassifier classes. Simply create an instance
and pass a binary classifier to its constructor.  For example, this code creates a multi‐
class classifier using the OvO strategy, based on a SGDClassifier:
            
            >>> from sklearn.multiclass import OneVsOneClassifier
            >>> ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
            >>> ovo_clf.fit(X_train, y_train)
            >>> ovo_clf.predict([some_digit])
            array([ 5.])
            >>> len(ovo_clf.estimators_)
            45

        >>> cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
        array([ 0.84063187, 0.84899245, 0.86652998])
It gets over 84% on all test folds. If you used a random classifier, you would get 10%
accuracy, so this is not such a bad score, but you can still do much better. For exam‐
ple, simply scaling the inputs (as discussed in Chapter 2) increases accuracy above
90%:

        >>> from sklearn.preprocessing import StandardScaler
        >>> scaler = StandardScaler()
        >>> X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
        >>> cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
array([ 0.91011798, 0.90874544, 0.906636 ])

Error Analysis:
- Here, we will assume that you have found a promising model and
you want to find ways to improve it.
- First, you can look at the confusion matrix. You need to make predictions using the
cross_val_predict() function, then call the confusion_matrix() function, just like
you did earlier:

        >>> y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
        >>> conf_mx = confusion_matrix(y_train, y_train_pred)
        >>> conf_mx
        array([[5725, 3, 24, 9, 10, 49, 50, 10, 39, 4],
         [ 2, 6493, 43, 25, 7, 40, 5, 10, 109, 8],
         [ 51, 41, 5321, 104, 89, 26, 87, 60, 166, 13],
         [ 47, 46, 141, 5342, 1, 231, 40, 50, 141, 92],
         [ 19, 29, 41, 10, 5366, 9, 56, 37, 86, 189],
         [ 73, 45, 36, 193, 64, 4582, 111, 30, 193, 94],
         [ 29, 34, 44, 2, 42, 85, 5627, 10, 45, 0],
         [ 25, 24, 74, 32, 54, 12, 6, 5787, 15, 236],
         [ 52, 161, 73, 156, 10, 163, 61, 25, 5027, 123],
         [ 43, 35, 26, 92, 178, 28, 2, 223, 82, 5240]])
That’s a lot of numbers. It’s often more convenient to look at an image representation
of the confusion matrix, using Matplotlib’s matshow() function:

            plt.matshow(conf_mx, cmap=plt.cm.gray)
            plt.show()

Let’s focus the plot on the errors. First, you need to divide each value in the confusion
matrix by the number of images in the corresponding class, so you can compare error
rates instead of absolute number of errors (which would make abundant classes look
unfairly bad):

        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums
Now let’s fill the diagonal with zeros to keep only the errors, and let’s plot the result:

        np.fill_diagonal(norm_conf_mx, 0)
        plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
        plt.show()
- Analyzing the confusion matrix can often give you insights on ways to improve your
classifier. Looking at this plot, it seems that your efforts should be spent on improving
classification of 8s and 9s, as well as fixing the specific 3/5 confusion. For example,
you could try to gather more training data for these digits. Or you could engineer
new features that would help the classifier—for example, writing an algorithm to
count the number of closed loops (e.g., 8 has two, 6 has one, 5 has none). Or you
could preprocess the images (e.g., using Scikit-Image, Pillow, or OpenCV) to make
some patterns stand out more, such as closed loops.

Multilabel Classification:
- Until now each instance has always been assigned to just one class. In some cases you
may want your classifier to output multiple classes for each instance


        from sklearn.neighbors import KNeighborsClassifier
        y_train_large = (y_train >= 7)
        y_train_odd = (y_train % 2 == 1)
        y_multilabel = np.c_[y_train_large, y_train_odd]
        knn_clf = KNeighborsClassifier()
        knn_clf.fit(X_train, y_multilabel)
- This code creates a y_multilabel array containing two target labels for each digit
image: the first indicates whether or not the digit is large (7, 8, or 9) and the second
indicates whether or not it is odd. The next lines create a KNeighborsClassifier
instance (which supports multilabel classification, but not all classifiers do) and we
train it using the multiple targets array. Now you can make a prediction, and notice
that it outputs two labels:\

        >>> knn_clf.predict([some_digit])
        array([[False, True]], dtype=bool)

- There are many ways to evaluate a multilabel classifier, and selecting the right metric
really depends on your project. For example, one approach is to measure the F1 score for each individual label (or any other binary classifier metric discussed earlier), then
simply compute the average score. This code computes the average F1 score across all labels:
        
        >>> y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
        >>> f1_score(y_multilabel, y_train_knn_pred, average="macro")
        0.97709078477525002
- This assumes that all labels are equally important, which may not be the case. In par‐
ticular, if you have many more pictures of Alice than of Bob or Charlie, you may want
to give more weight to the classifier’s score on pictures of Alice. One simple option is
102 | Chapter 3: Classification 4 Scikit-Learn offers a few other averaging options and multilabel classifier metrics; see the documentation for more details. to give each label a weight equal to its support (i.e., the number of instances with that target label). To do this, simply set average="weighted" in the preceding code.

Multioutput Classification:
- It is simply a generalization of multilabel classification where each label can be multiclass (i.e., it can have more than two possible values).
- To illustrate this, let’s build a system that removes noise from images. It will take as
input a noisy digit image, and it will (hopefully) output a clean digit image, repre‐
sented as an array of pixel intensities, just like the MNIST images. Notice that the
classifier’s output is multilabel (one label per pixel) and each label can have multiple
values (pixel intensity ranges from 0 to 255). It is thus an example of a multioutput
classification system.
- Predicting which pixels need to be less noisy:

        knn_clf.fit(X_train_mod, y_train_mod)
        clean_digit = knn_clf.predict([X_test_mod[some_index]])
        plot_digit(clean_digit)

## Lecture 5 Videos:

### Video 5.1: Classification Problems:
<img width="305" alt="image" src="https://github.com/user-attachments/assets/9cb29f1f-fe08-4c5f-8635-c93cd43a50d8">
- Instead of a real-value response, classifciation assign c to category ot class.
    - Regression: For pair (x, y), y is the response value of x
    - Classification: For pair (x,y), y is the class label of x
- INput: Measurement x1, ....., xn, is an input sapce
- Output:  Discrete output is composed of k possible classes:
    - y = {-1, +1} or {0,1} is called a binary classification problem
    - y = {1, ....., K} , is called a ,multiclass classification problem

Binary Classification (2-class):
- Simplify the problem into a 5 detector.

Training a binary classifier:
<img width="250" alt="image" src="https://github.com/user-attachments/assets/74e4a4db-510f-4165-ae63-4af3d170441c">
- able to handle large data sets


Performance Measures
-  it is a lot harder to do this with classifiers. We have different options.

### Video 5.2: Metrics for Classification Models
<img width="299" alt="image" src="https://github.com/user-attachments/assets/fc945252-9f0f-4224-b009-e3508adaf3a6">
- visual example of recall vs precision

Confusion Matrix:
- helps gain insight on the performance of your classifier:
<img width="308" alt="image" src="https://github.com/user-attachments/assets/c6295a36-a07b-4881-98e0-77a2842e141a">
<img width="258" alt="image" src="https://github.com/user-attachments/assets/0dde7362-88d9-4767-b185-6c7c4206bdf6">

More concise metrics (than confusion matrix):
<img width="307" alt="image" src="https://github.com/user-attachments/assets/c3d8029a-0b44-4736-9c21-2f77af8bded8">

Boy who called wolf analogy?
- Recall:
  - Out of all the times the wolf comes, how many times did he get it right?
  - Intuition: Did the model miss many wolves?
- Precision:
  - Of all the times the boy cried wolf, how many times did get it right?
  - Intuition:Did the model cry wolf too often?

Even more concise metic: F-1 Score
- harmonic mean between precision and recall
<img width="311" alt="image" src="https://github.com/user-attachments/assets/1443ca5d-6946-4bfa-a1ce-0a0cc96178a4">
- most optimized when both precision and recall are the same

The perfecr predicions. (we wish)
<img width="300" alt="image" src="https://github.com/user-attachments/assets/58933060-741e-4db0-aaa1-97c6c4f921b1">

### Video 5.3: The Trade off between Precision and Recall
<img width="316" alt="image" src="https://github.com/user-attachments/assets/4a577010-145b-48f5-9377-a2909dfcba1a">
<img width="299" alt="image" src="https://github.com/user-attachments/assets/0fb44b7b-392b-463d-8b07-705287a56857">


The ROC Curve:
- axis are a little different
<img width="308" alt="image" src="https://github.com/user-attachments/assets/1a31ef1c-0ec4-449b-bdce-f53fed7c9fd9">
Area Under the Curve (AUC):
- literaly j the area under ROC Curve
- You can use the ROC curve to compare the difference between different classifiers.

## Lecture 6 Videos 

### Video 6.1: Logistic Regression for Classificiation
<img width="311" alt="image" src="https://github.com/user-attachments/assets/a0a0f860-699e-49d7-bc45-f36de76754f8">
<img width="311" alt="image" src="https://github.com/user-attachments/assets/4deb3d91-b0be-4e99-9c13-4a53d6782e43">
- this is nonlinear
<img width="290" alt="image" src="https://github.com/user-attachments/assets/55038b44-c2e7-4cc1-be32-f9596a905ba0">

Decision Boundary: Linear Example:
<img width="316" alt="image" src="https://github.com/user-attachments/assets/65f6f60c-29e6-41e3-9192-eb38dc4547c4">

Decision Boundary: Non Linear Example

<img width="316" alt="image" src="https://github.com/user-attachments/assets/eaa301b0-691b-40e3-8c56-c8da0b948b11">

Loss/Cost/Error Function:
- need to find a new lost function, so that we can still use gradient descent.
<img width="316" alt="image" src="https://github.com/user-attachments/assets/b2f76bd9-79fa-4473-b568-8168f3eee9f2">
<img width="415" alt="image" src="https://github.com/user-attachments/assets/da0353a9-c703-44b3-a52e-ec85306ce9de">
- when y =1
<img width="415" alt="image" src="https://github.com/user-attachments/assets/01e48b40-680f-42a6-9e00-35327709107a">
- when y =0

Combining the two cases: The LOG LOSS FUNCTION
<img width="623" alt="image" src="https://github.com/user-attachments/assets/64de417c-05b3-44dd-a982-4d35659d3d53">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/7d1bde22-c14c-40f2-9321-b3f843cde411">
- vectoried version.

### Video 6.2: Gradient Descent For Logisitc Regression
- There is no known closed form to compute the value of the parameter that minimizes the cost function (there is no "Normal Equation" for the Logistic Regression)
- The new cost function is convex, so Gradient Descent is quarennted to find the global minimum.

Gradient Descent Formulation:
<img width="305" alt="image" src="https://github.com/user-attachments/assets/29197153-d471-46ae-b7fc-83d7964502ea">
<img width="227" alt="image" src="https://github.com/user-attachments/assets/4069c360-2c77-476a-bdd8-eb002f0bbe74">
<img width="314" alt="image" src="https://github.com/user-attachments/assets/05d1a918-2d10-4a36-9d6d-693e27ab14b4">

Final:
<img width="276" alt="image" src="https://github.com/user-attachments/assets/3fa424b8-aa7c-40ee-80d4-19bf06f3867e">
- looks really similar to linear regression

Gradient Descent Step:
<img width="312" alt="image" src="https://github.com/user-attachments/assets/f3cd9d81-44cd-4083-a07d-bb8495c1d1c5">
- we can compute the gradient of the model at every single step.
- for the gradient update you do this:
<img width="205" alt="image" src="https://github.com/user-attachments/assets/c3bb96b3-54fc-49c8-bfde-dcff2dfbb3e1">
- batch you use all of training for X and y, minibatch you use a subset for X and y, and for the last one you do one at a time. 

# Module 7: Unsupervised Learning

## Reading Chapter 9

## Lecture 7 Videos: Clustering Methods HAC and k-means

### Video 7.1: Similarity Measures
- its hard to describe what the similarities are. especially for a computer
- its easier to think of it in terms of the distance between vectors.

Distance Measures:
<img width="491" alt="image" src="https://github.com/user-attachments/assets/e30a4a6c-7bb0-4fe9-a7a0-ef437f90c246">
- Minkowski Metric
  - <img width="347" alt="image" src="https://github.com/user-attachments/assets/3dab1849-14ec-46e0-b4bf-b4cfefacf874">
  - <img width="368" alt="image" src="https://github.com/user-attachments/assets/316ce1dd-b98b-41f9-a32e-f38708138415">
  - Example:
    - <img width="336" alt="image" src="https://github.com/user-attachments/assets/9207f2c7-2815-4466-b1df-43df8756634a">
- Hamming Distance:
  - <img width="505" alt="image" src="https://github.com/user-attachments/assets/9a901449-c187-4498-8a20-f8e188237a60">
- Edit/Transform Distance:
  - To measure the similarity between two objects, transform one object into the the other, and measure the effort it takes.
  - <img width="467" alt="image" src="https://github.com/user-attachments/assets/f14c6e5b-8e4d-4742-ba1c-9960695518f3">


### Video 7.2: Hierarchical Clustering (HAC)
There are 2 different types of Clustering Algs:
- Hierarchical alg:
  - bottom up- agglomerative
  - top-down: divisive
- Partitional Algs:
  - Usually start with a random (partial) partitioning
  - Refine it iteratively (K-means)

Hierarchical Clustering:
-Dendogram:
<img width="583" alt="image" src="https://github.com/user-attachments/assets/8ce0bee1-42b8-42b8-a1d8-6becf0248d1d">
- crazy number of dendogram combos
- not a very scalable way to do things
- start with each item on its own cluster, and then find the next best item to join together into a cluster:
- Bottom Up approachL
  - begin with a distance matrix which contains the disteances between each pair of objects.
  - You consider all possible merges from the base case.
  - You simply pick the best. Its greedy in a way.
  - <img width="340" alt="image" src="https://github.com/user-attachments/assets/52e1e7e4-5f4b-446a-85a1-327fe119ecc7">
  - <img width="262" alt="image" src="https://github.com/user-attachments/assets/72d52ff3-a16f-4522-a334-03314274d8ce">
  - <img width="583" alt="image" src="https://github.com/user-attachments/assets/fc1116a9-c17f-460f-bd85-37c41ae2c5bd">

- Numerical Example:
  - <img width="383" alt="image" src="https://github.com/user-attachments/assets/1bbc157d-6d8d-4cfb-b5a3-874a42a42ae1">

- Complexity of HAC:
  - In the first iteration, it takes a lot to compute those values O(m^2 n)
  - Compute the distance between the most recently created cluster and all existing clusters O(mn)
  - Maintain a min-heap to find the smalled pair O(mn logm)
  - Since step 2,3 needs to be done in each subsequence. merging O(m^2 n log m)
  - It does not scale well O(m^2 n log m) with local optima.
### Video 7.3: DBSCAN and other clustering methods
- DBSCAN:
  - Density based spatial clustering of applications with noise is another clustering alg which is based on loca, density estimation
  - Simple yet powerful alg which allows identification of arbitrary shapes
  - Works well if all the clusters are desnse engouh and well seperated by low-density region.
  - <img width="586" alt="image" src="https://github.com/user-attachments/assets/e9b3c682-6b27-4cfc-97d6-d455e321d9be">
  - <img width="574" alt="image" src="https://github.com/user-attachments/assets/335cc0d5-1659-4f2a-a09c-a7169c0322a2">
- Complexity:
  - has two hyperparams (eps and min_samples)
  - computationsal complex. is roughly O(mlogm)
  - if esp is large, it may require up to O(m^2) memory
<img width="563" alt="image" src="https://github.com/user-attachments/assets/274292cf-d411-4e60-95c4-f511f94c8f26">
- (not tested on these algs)

### Slides from Clustering:
<img width="437" alt="image" src="https://github.com/user-attachments/assets/3f3ffa63-af3e-49eb-9f2d-24d447c92df5">

Partition Clustering (K-means)
- construct a partition of m objects into a set of k clusters.
- user has to specify the desired number of clusters
- partitioning alg:
  - <img width="418" alt="image" src="https://github.com/user-attachments/assets/854abacc-d8f6-488c-ace1-225ea0cb3ba4">
  - <img width="434" alt="image" src="https://github.com/user-attachments/assets/3fe7cf68-c682-4d28-a48a-7ca6af7702cc">
- <img width="445" alt="image" src="https://github.com/user-attachments/assets/bfbe50f2-54c0-455e-b0e2-f6b18fe5c5f4">
- <img width="436" alt="image" src="https://github.com/user-attachments/assets/5d21bd95-5c22-4e6f-be06-cc7426ed75f9">
- <img width="431" alt="image" src="https://github.com/user-attachments/assets/a67d0c0c-cd04-4b38-bccf-e6b55d899ddb">
- <img width="443" alt="image" src="https://github.com/user-attachments/assets/a58e2450-3d7c-4b10-ad89-380fc6820db9">
- <img width="437" alt="image" src="https://github.com/user-attachments/assets/720c9378-ea3f-49ed-9904-4b5018724692">
    - need to know this


# Module 8: Decision Tree Learning

## Reading Chapter 6:

## Lecture 8 Videos:

### Video 8.1: Decision Trees
- using a tree structure to split data into groups
- <img width="603" alt="image" src="https://github.com/user-attachments/assets/37b3bf59-819f-4329-aca8-062942cfcd02">
- the number one thing is determining what feature to split on and what is the threshold.
- <img width="571" alt="image" src="https://github.com/user-attachments/assets/49a84039-bf88-4b0d-bc17-6fc04643d90a">
  - Gini Index:
      - <img width="348" alt="image" src="https://github.com/user-attachments/assets/1d800ded-5b61-4844-aaeb-84133fb8ff3b">
      - <img width="500" alt="image" src="https://github.com/user-attachments/assets/86811725-643a-436b-964e-03b97538ea1a">
          - this is the gini index for that top node
          - orange is 0, o.168, 0.0425
          - lower gini index, it means your nodes are more pure. 
          - as u move down the branches, the gini index decreases as you move down. You want your leaf nodes to be pure. 
  - Decision Boundary:
      - <img width="673" alt="image" src="https://github.com/user-attachments/assets/2c51a3de-a39e-46d2-ad51-0c43564f09d1">
      - <img width="574" alt="image" src="https://github.com/user-attachments/assets/d271235b-f04b-4873-8014-c29b73567967">
      





### Video 8.2: Entropy and Information Gain
Entropy- 
- a measure of disorderness. it is zero when well-order and identical
- A set's entopy is zero when it contains instances of only one class.
- Entrop is H st.
- <img width="610" alt="image" src="https://github.com/user-attachments/assets/8e5bfcf1-a64b-4956-9592-0736ece61d06">
- gini tends to isolate that largest class
- <img width="593" alt="image" src="https://github.com/user-attachments/assets/080467cc-ee24-48bf-ac29-9844e83f416d">
v<img width="585" alt="image" src="https://github.com/user-attachments/assets/eb34ce47-4198-4d15-8cc0-31e08ca45ab6">

Information Gain:
- <img width="545" alt="image" src="https://github.com/user-attachments/assets/02456bad-a7c6-4061-b573-92455f100839">

- trees need to be somewhat small

<img width="593" alt="image" src="https://github.com/user-attachments/assets/3add5aac-0595-4ae3-b986-b53417591f9b">


Measure teh resucrtion in entropy of Y from knowing Xm we use information Gain:




Entropy Interpretation.


### Video 8.3: Limitations and Applications of Decision Trees
<img width="607" alt="image" src="https://github.com/user-attachments/assets/16475675-4bc5-4282-ab9c-0c9a3a306806">


# Module 10: Analogical Learning

## Reading Chapter 5:

## Lecture 10a Videos:

### Video 10.1

### Video 10.2

### Video 10.3 

## Lecture 10b Videos:

### Video 10.4

### Video 10.5

### Video 10.6










