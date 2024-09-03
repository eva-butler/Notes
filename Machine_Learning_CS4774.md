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
          - hypothesis space: the set of functions that the learning data algortihm us allowed to select as being the solution
             - 
