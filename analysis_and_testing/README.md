# Thought-Process

<p><b>Goal</b>: The primary goal is to build an automated machine learning pipeline that can predict the risk of loan applications. This pipeline should handle data ingestion, preprocessing, model training, evaluation, and deployment seamlessly.</p>

---

## Data Understanding (Notebook):

### Loan Dataset

<ul>
    <li> Need to deal with missing values </li>
    <li> Replacements mentioned in data_understanding notebook </li>
    <li> Need to deal with outliers </li>
    <li> Encoding of categorical variables </li>
    <li> Need to scale numerical features </li>
    <ul> 
    <li> Feature Engineering </li>
        <ul>
        <li> Extract Year and Month from Application Date </li>
        <li> payFrequency should have ordinal numerical representaiton </li>
        <li> Ratio between lead cost and loan amount </li>
        </ul>
    </ul>
</ul>

### Target Variable
<ul>
    <li> Change to binary values of high risk and low risk </li>
    <li> Leads to unbalanced dataset which will be internally handled using LightGBM </li>
</ul>

### Payment Dataset
<ul>
    <li> Wanted to group by loanId and extract useful data </li>
    <li> Only 40k unique values in loanId column </li>
    <li> Too little, not useful in the large model </li>
</ul>

---

## Data Cleaning (Notebook):

### Null Values
<ul>
    <li> Drop the columns that are not requried and have too many missing values </li>
    <li> Dropped ('loanId', 'originatedDate', 'fpStatus', 'clarityFraudId') </li>
    <li> Also drop 'anon_ssn' because after it is encoded, not useful enough when base learners are decision trees </li>
    <li> Replaced null values of ('payFrequency', 'apr', 'nPaidOff', 'loanAmount', 'state') with suitable values </li>
    <li> Dropped the rows with null values for the target variable </li>
</ul>

### Outliers
<ul>
    <li> Check outliers of 'loanAmount' column </li>
    <li> Considering the 25-75 percentile has around 50,000 outliers </li>
    <li> Removed the outliers </li>
    <li> Similarly, dealt with outliers of 'originallyScheduledPaymentAmount' and 'apr' columns </li>
    <li> Lost around 76,000 datapoints after removal of outliers. That is around 16% of data </li>
</ul>

### Encoding Categorical Values
<ul>
    <li> Need to deal with only categorical column. Rest will be handled by LightGBM automatically </li>
    <li> Dealing with 'payFrequency' column becaause it requires ordinal encoding </li>
    <li> Ordinal map with frequent payments as high number and non-frequent payments as low numbers </li>
    <li> Mapped and added to the dataframe </li>
</ul>

### Feature Engineering
<ul>
    <li> Extracted Year and Month data from 'applicationYear' column </li>
    <li> Created a feature exploring ratio between the lead-cost and the loan-amount </li>
    <li> Searched online for season related data in USA (based on 'state' column) </li>
    <li> Extracted seasonal information from 'applicationMonth' column </li>
</ul>

### Target Variable
<ul>
    <li> Selected a few values of 'loanStatus' column to label as high-risk </li>
    <li> Create a new feature which will be the target variable of the model </li>
    <li> The new feature describes the data points as high-risk or low-risk </li>
    <li> The 'loanStatus' column will be deleted </li>
</ul>

### Scaling Numerical Features
<ul>
    <li> 'loanAmount' and 'apr' do not have proper normal distribution </li>
    <li> But maximum datapoints are at the center of the graph </li>
    <li> MinMax Scaling will be used to scale these values </li>
    <li> The 'originallyScheduledPaymentAmount' column has normal distribution </li>
    <li> Normalizer will be used to scale these datapoints </li>
</ul>

---

## EDA (Notebook):

### Prerequisites
<ul>
    <li> Change the datatypes in accordance with numerical and categorical data </li>
    <li> Suggest points of improvement if the model underperforms in its first run </li>
</ul>

### Observations from EDA
<ul>
    <li> Loan Amount has 0 as its minimum </li>
    <li> The dataset has very low correlation </li>
    <li> Due to low correlation, we need non-linear solutions and ensemble solutions </li>
    <li> The suggested LightGBM model is an ensemble solution with boosting and also is a non-linear solution as the base learners of LightGBM are decision trees </li>
    <li> The target variable is unbalanced, need to do stratified splitting of the data for train-test split </li>
    <li> POI: Most of the numerical values do not have a normal distribution </li>
    <li> POI: Most of the categorical values are unbalanced </li>
</ul>

---