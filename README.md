# Klarna Take Home Assignment

This is the repository containing the solution for the take home assignment.

The problem is of credit default prediction. As is typical in such problems, the target variable 'default' was highly imabalanced with only 1.5% defaults. 
Since I had to create an end-to-end to solution in a limited amount of time, I didn't spend too much time on optimising all possible parts of the process. 

## Approach 

- A very basic, high-level EDA was performed to check between-feature correlation, and also correlation of all features with the target variable. 

- Then, I looked at the missing values and the pattern of missing values. It looked like the missing values were not missing completely at random, and they had some pattern to them. Since I didn't have more access to the domain understanding of each feature, I decided not to fill up the nulls. Instead, I used XGBoost for making the prediction, which accepts datasets with null values. 

- For the class imbalance, I ended up going ahead with Random Undersampling, where the majority class was undersampled to be of equal size to the minority class making it a 50-50 split. The reason for going ahead with this was that the XGBoost seemed to overfit the data the least here.

## Evaluation

- Since this was a class-imbalanced problem, I decided to do away with a simple accuracy metric to judge the efficacy of the model. 
- Instead, I looked at Precision, Recall and F1 Score. 
- These values can be optimised by choosing the most optimal probability threshold for the model prediction probability. 
- The probability threshold can be chosen based on the business use-case. Here, since we're detecting credit defaults, we might want to prioritise precision over recall as we don't want to wrongly predict default multiple times. This would lead to poor customer experience. 
This would mean setting a probability threshold that's on the higher side (higher than the 0.015 ratio of the positive class)

## Deployment

- The model and other related variables were saved as .pkl files and then the model was served as an API using the FastAPI framework.

- Docker has been used to containerize the whole prediction ecosystem. 

- The model has been deployed using an AWS EC2 instance. 

## API Endpoint

- The API endpoint of the above model is http://35.171.30.75:8000/predict
Since FastAPI was used, http://35.171.30.75:8000/docs can be access to get more information about the features, and we can also make a post request within the browser. 
- Making a post request with a JSON containing all the features (that are present in the test set) will return a prediction for that particular customer. 
