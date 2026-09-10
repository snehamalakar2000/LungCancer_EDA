# Lung Cancer EDA and Machine Learning

## Overview

In this project, I explored a lung cancer patient dataset using pandas and Matplotlib, then tested a simple logistic regression model to predict survival.

## Data Inspection

I used `head()`, `info()`, and `describe()` to get familiar with the dataset. I also checked for duplicates and unique patient IDs.

## EDA

I converted the `Survived` column into a numeric variable so I could calculate survival rates.

When I grouped patients by cancer stage, I found a clear pattern:

- Stage I: ~69.8% survival
- Stage II: ~50.7%
- Stage III: ~17.4%
- Stage IV: ~5.5%

I then filtered the data to compare Stage I and Stage IV patients and looked at smoking status. Stage IV had a higher share of current smokers, while Stage I had more patients who had never smoked.

I also created a scatter plot of tumor size vs. survival months. The plot showed a negative relationship, with larger tumors generally associated with shorter survival times.

## Machine Learning

I used logistic regression to predict whether a patient survived. My first model used:

- Cancer stage
- Tumor size
- Age

I split the data into training and test sets and trained the model on 80% of the data. The model achieved about **75% accuracy** on the test set.

## Main Takeaways

Cancer stage showed the strongest survival pattern in my EDA. Tumor size also appeared to have a negative relationship with survival time. The logistic regression model gave me a basic starting point for predicting survival using patient characteristics.
