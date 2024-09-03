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
