# Loan Risk Prediction Model

## Purpose of the Project

The purpose of this project is to develop a robust pipeline capable of training a predictive model specifically for assessing loan risk. The significance of such a model is it can predict which loans carry high risk of defaulting or have late payments. By doing so, it enables companies offering these loans to reduce the possible losses brought on by defaults and late repayments, improving their overall financial stability.

Note: The pipeline is produced after initial testings using the dataset. All the code and notes of the initial testing and data understanding can be found in `analysis_and_testing` folder of this project. It also contains detailed thought-process of understanding and using the datasets, the justifications for data cleaning and takeaways of EDA.

## Overview

The pipeline uses LightGBM Classification model for classification of risk. This project focuses on using a dataset generated within a production environment by a loan provider company. The pipeline's design has been customized to accommodate the unique format and characteristics of this dataset, ensuring smoooth integration and optimal performance. The ultimate goal is to equip the financial institution with an effective risk assessment tool so they can minimize the financial risks related to loan provision and make well-informed decisions.

The results shown in this report are using confusion matrix. The model correctly predicts 95,744 datapoints and gets wrong the other 12,246. This shows 88.6% accuracy in the model.

## Using the Training Pipeline

To download the necessary libraries, use the command given below in a python environment of your choice.

    pip install -r requirements.txt

Run the first five code blocks in the `pipeline.ipynb` notebook. These blocks of code initialize the pipeline necessary for training the model. Pass the dataframe `df` as input to the trianing pipeline which outputs a trained model. The process also displays the confusion matrix of the testing dataset.

    model = pipe_training.fit_transform(df)

## Inference using the Trained Model

Pass the inference dataset `inf_df` through the feature transform pipeline to transform the dataset for prediction.

    inf_df = pipe_data_feature_transform.fit_transform(inf_df)

Use the below code to predict the risk factor of all the inference datapoints. The resulting value is a list object with all the predictions.

    pred = model.predict(inf_df)

# Pipeline Architecture

- Data Preprocessing the Target and Outliers (for Training only)
    - Transforming target variable
        - Drop rows with Null values in target variable
        - Transform the target values to binary values (High Risk and Low Risk)
        - Drop the `loanStatus` column
        - Use `binaryRiskStatus` as the target column
        - Encode the target column to 0 (Low) and 1 (High)
    - Remove outliers in the dataset

- Data Preprocessing for Features (for Training and Inference)
    - Dropping data
        - Drop unnecessary columns
    - Imputing Null values
        - With most-frequent value
        - With average value
    - Feature Engineering
        - Year and Month from `applicationDate`
        - Product of `loanAmount` and `leadCost`
        - Seasons column using Month
    - Scaling of Data
    - Categorical Encoding
        - Ordinal Encoding for `payFrequency` and other data
        - One hot encoding for binary categories
- Model Training
    - Split to features and target
    - Split to train and test
    - Train the model on training data
    - Show confusion matrix on testing data

# FYI

## Clarifications

- Why is there reliance on static data features (fixed data format)?

This is because in a deployment environment, the dataset produced from a company will be produced in the same format unless changed specifically. This data can be used to further train the model. 
The performance improvement of the model for specific transformation of data features is deemed to be more valuable than the dynamic data processing for model training.

---

- Downside of the Results

The downside of the model is that it predicts 9,941 risky loans as not risky (false negatives) and predicts 2,305 non-risky loans as risky (false positives). More false negatives is a bad result for loan risk prediction model because false positives are more preferrable in this case.

---

## Future Improvements

- Hyperparameter Optimization

Use hyperparameter optimization library such as `optuna` to further improve the model with the best possible parameters for the highest accuracy model. 

You can also use cross validation techniques which is used to choose the best model out of all the cross validation datasets.

---

- Inference Pipeline

Due to time constraints, the proposed solution is simple inference. A recommended update to the model would be creating a pipeline to do the following:

    - Process the features of the inference dataset 
    - Predict using the trained model
    - The predictions are added as the new column for the inference dataset
    - The new dataset with attached prediction will be the output of the pipeline

---

- Dependency and Package Management

A widely used solution like `poetry` would be suggested for a deployemnt environment. The `requirements.txt` is used in this case as there are simple dependencies and use of python notebooks where selecting the required environment is a bit more complicated using poetry.

---

- Further Training the Model

LightGBM is a model that is extremely efficient and very quick to train. If other boosting models are used like XGBoost and there are millions of data points, then the training of the model will have to have further training for quicker deployment of models. A further training pipeline would be recommended.

---