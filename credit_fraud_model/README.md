# Kaggle credit card fraud 

## Tl;dr

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.
* [Kaggle Link](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Context

The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

## Features
It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## Model Preprocessing 

Since this is highly unbalanced dataset with positive class only 0.17% and for better model performance,I chose to do oversampling positive class using SMOTE algorithm (Synthetic Minority Oversampling Technique). At a high level, SMOTE creates synthetic observations of the minority class (bad loans) by:
* Finding the k-nearest-neighbors for minority class observations (finding similar observations) Randomly choosing one of the k-nearest-neighbors and using it to create a similar, but randomly tweaked, new observation.
There’s no need (and often it’s not smart) to balance the classes, but it magnifies the issue caused by incorrectly timed oversampling.
* After trying different upsampling ratio, when fraud class is 10% of negative class, the model performed the best in the test set. (In theory we can add this step in the training pipeline and use grid search to find the best ratio, but since I chose XGboost algorithim and my verison of XGboost can't build the pipeline, thus I just do some manual check )

## Model Selection

Train respectively using Logisitic Regression, Random Forest and XGBoost. The performance metrics are as below:

```
LogisticRegression : Train AUPRC 92.54% ; Valid AUROC 56.30%; ROC 93.33% ; training time 1.0 seconds
RandomForestClassifier : Train AUPRC 89.15% ; Valid AUROC 67.66%; ROC 93.35% ; training time 51.0 seconds
XGBClassifier : Train AUPRC 99.84% ; Valid AUROC 71.76%; ROC 92.33% ; training time 60.0 seconds

```
<img width="406" alt="train_plot" src="https://user-images.githubusercontent.com/50553507/103152389-ac314800-4787-11eb-8248-d161837f7c91.png">

Since this is super unbalanced set, fraud only is 0.17% (Baseline of AUPRC is 0.17%) among all the instances. Thus I chose AUPRC here as performance measure, otherwise ROC would be misleading due to always minor false positive. As can seen from PR AUC curve, XGBoost performed the best with 71% PR AUC compared with Logistic Regression and Random Forest models.
Therefore I will proceed XGBoost with model hyperparameter tuning to find the optimal thresholds for hard rejection and manual reveiw.

## Hyperparameter tuning

Use average_precision score to select the best performing hyperparamters

```
random_search = RandomizedSearchCV(XGBClassifier(),{
        'min_child_weight': [1, 5, 10],
        'gamma': [1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'n_estimators':[10,20,50,100]
        }, scoring='average_precision',cv=3,n_iter=6,random_state=41)

```

**Best score**
0.9996710508802215

**Best hyperparameters**
{'subsample': 0.6, 'n_estimators': 100, 'min_child_weight': 10, 'max_depth': 5, 'gamma': 2, 'colsample_bytree': 0.6}

Retrain the model using the best hyperparameters and full train set

* AUPRC score of train set:0.998
* AUPRC score of test set:0.715

### Feature Importance

Based on feature importance, we can see there are no feature dominated the variance, the top important features are v13,v3 and time

<img width="395" alt="feature_importance" src="https://user-images.githubusercontent.com/50553507/103152364-712f1480-4787-11eb-9abf-ba6791309c16.png">

```
data['avg_score']=data['score']/data['score'].sum()
data.sort_values(by='avg_score',ascending=False).head(5)

```

|     | score | avg_score |
|-----|-------|-----------|
| f14 | 91    | 0.066863  |
| f4  | 85    | 0.062454  |
| f0  | 79    | 0.058046  |
| f7  | 57    | 0.041881  |
| f17 | 57    | 0.041881  |


## Threshold Definition

Below we will define thresholds for hard reject and manual review by considering recall/precision and financial impact

<img width="386" alt="test_plot" src="https://user-images.githubusercontent.com/50553507/103152392-b81d0a00-4787-11eb-810d-f5226bf0826e.png">

Look at the precision - recall and other metrics when thresholds under 0.995,0.99,0.98,0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1

|    | threshold | precision |   recall | transaction_perc | financial_save_perc | financial_loss_perc |
|---:|----------:|----------:|---------:|-----------------:|--------------------:|--------------------:|
|  0 |     0.995 |  1.000000 | 0.622449 |         0.001071 |            0.562548 |            0.000000 |
|  1 |     0.990 |  0.985915 | 0.714286 |         0.001246 |            0.603140 |            0.000041 |
|  2 |     0.980 |  0.986486 | 0.744898 |         0.001299 |            0.609469 |            0.000041 |
|  3 |     0.950 |  0.975000 | 0.795918 |         0.001404 |            0.680443 |            0.000049 |
|  4 |     0.900 |  0.963855 | 0.816327 |         0.001457 |            0.680574 |            0.000050 |
|  5 |     0.800 |  0.911111 | 0.836735 |         0.001580 |            0.770263 |            0.000207 |
|  6 |     0.700 |  0.881720 | 0.836735 |         0.001633 |            0.770263 |            0.000556 |
|  7 |     0.600 |  0.854167 | 0.836735 |         0.001685 |            0.770263 |            0.000869 |
|  8 |     0.500 |  0.854167 | 0.836735 |         0.001685 |            0.770263 |            0.000869 |
|  9 |     0.400 |  0.828283 | 0.836735 |         0.001738 |            0.770263 |            0.000983 |
| 10 |     0.300 |  0.761468 | 0.846939 |         0.001914 |            0.770263 |            0.001190 |
| 11 |     0.200 |  0.666667 | 0.877551 |         0.002265 |            0.770526 |            0.001609 |
| 12 |     0.100 |  0.520710 | 0.897959 |         0.002967 |            0.777044 |            0.002849 |

Based on the evaluations of some metrics: the model performance ( precision&recall ), num_transactions_percent,financial_save_percent (what percent of money we are able to save) and financial_loss_percent (only for deciding hard reject threshold, the percent of incorrect rejection to legit customers) under each threshold, we decide which threshold chosen for hard reject and which one chosen for manual review

* As we can see from table above, when model threshold>=0.99, precision is 98.6% with 71% of recall rate, which will hard reject 60% of fraudulent transactions with almost 0% loss from incorrectly rejecting legit customers, therefore if we set model threshold 0.99 as hard reject threshold, we are potentially able to block 56% of fraudulent traffic with neglectable loss from incorrectly blockling legit customers

* Now let's find the optimal threshold for manual review ( those transaction will be reviewed by fraud analysts ), since fraud analysts have limited capacities, in reality better based on the capacities of analyst team and transaction base to decide the optimal threshold for the team. Here I assume there is enough capacities from operation team and we can see the financial save percent doesn't change dramatically after 0.8. Therefore if choosing 0.8 model score as threshold for manual review, we can prevent 21% more financial loss on top of model hard reject.




































