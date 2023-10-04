### Scenario
Sailfort Motors are currently having a high rate of turnover. Since Sailfort Motors invest in recruiting, training, and upskilling their employees, the impact of high turnover rate is costly in the financial sense. The HR department were tasked by the leadership team of Sailfort Motors to collect sample data from the employees on what might be the cause of high turnovers. The leadership also tasked you to analyze the survey data and design a model to predict employee departure.

Note: Turnover data includes both employees who choose to quit their job and employees who are let go.

The task also involves identifying the factors that contribute to employee departure. The model will be beneficial to the company by increasing retention rate and job satisfaction of employees.

#### Issue/Problem
Sailfort Motors seeks to improve employee retention.
Response
Since the variable to be predicted is categorical, a logistic regression or tree-based machine learning model will be used.
Impact
The model will help predict employee departure and the factors that causes it. The HR can also devise a plan to prevent departures and improve employee retention.
Familiarization of the HR Dataset
The dataset was provided by Google Certificates, but can also be downloaded in Kaggle. The dataset contains 14,999 rows and 10 columns. Shown below are the variables used in the dataset and its corresponding description.

Variable |Description |
-----|-----| 
satisfaction_level|Employee-reported job satisfaction level [0–1]| 
last_evaluation|Score of employee's last performance review [0–1]| 
number_project|Number of projects employee contributes to| 
average_monthly_hours|Average number of hours employee worked per month| 
time_spend_company|How long the employee has been with the company (years)| 
Work_accident|Whether or not the employee experienced an accident while at work|
left|Whether or not the employee left the company| 
promotion_last_5years|Whether or not the employee was promoted in the last 5 years| 
Department|The employee's department|
salary|The employee's salary (U.S. dollars)

### Import the necessary packages
```
## For data manipulation
import numpy as np
import pandas as pd

## For data visualization
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

## For displaying all columns in the dataframe
pd.set_option('display.max_columns',None)

## For data modeling
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

## For metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve
from sklearn.tree import plot_tree

# For saving models
import pickle

# Miscellaneous
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
```
### Load Dataset
```
df0 = pd.read_csv("HR_comma_sep.csv")
## Display the first 10 rows of data
df0.head(10)
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>Department</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.41</td>
      <td>0.50</td>
      <td>2</td>
      <td>153</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.10</td>
      <td>0.77</td>
      <td>6</td>
      <td>247</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.92</td>
      <td>0.85</td>
      <td>5</td>
      <td>259</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.89</td>
      <td>1.00</td>
      <td>5</td>
      <td>224</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.42</td>
      <td>0.53</td>
      <td>2</td>
      <td>142</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>

### Data Exploration
#### Basic Information about the Data
```
## Basic information about the data
df0.info()
```
```
RangeIndex: 14999 entries, 0 to 14998
Data columns (total 10 columns):

 #   |Column                 |Non-Null Count  |Dtype  |
-----|-----------------------|----------------|-------|  
 0   |satisfaction_level     |14999 non-null  |float64|
 1   |last_evaluation        |14999 non-null  |float64|
 2   |number_project         |14999 non-null  |int64  |
 3   |average_montly_hours   |14999 non-null  |int64  |
 4   |time_spend_company     |14999 non-null  |int64  |
 5   |Work_accident          |14999 non-null  |int64  |
 6   |left                   |14999 non-null  |int64  |
 7   |promotion_last_5years  |14999 non-null  |int64  |
 8   |Department             |14999 non-null  |object |
 9   |salary                 |14999 non-null  |object |

dtypes: float64(2), int64(6), object(2)
memory usage: 1.1+ MB
```
Based on the basic information about the data, there are 14,999 non-null objects for all columns, equal to the total number of rows, which means that there are no null values in the dataset. The corresponding datatype for each columns are also shown.

#### Descriptive Statistics about the Data
```
df0.describe()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.612834</td>
      <td>0.716102</td>
      <td>3.803054</td>
      <td>201.050337</td>
      <td>3.498233</td>
      <td>0.144610</td>
      <td>0.238083</td>
      <td>0.021268</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.248631</td>
      <td>0.171169</td>
      <td>1.232592</td>
      <td>49.943099</td>
      <td>1.460136</td>
      <td>0.351719</td>
      <td>0.425924</td>
      <td>0.144281</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.090000</td>
      <td>0.360000</td>
      <td>2.000000</td>
      <td>96.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.440000</td>
      <td>0.560000</td>
      <td>3.000000</td>
      <td>156.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.640000</td>
      <td>0.720000</td>
      <td>4.000000</td>
      <td>200.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.820000</td>
      <td>0.870000</td>
      <td>5.000000</td>
      <td>245.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>310.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

#### Rename Columns
Standardize column names into snake_case ,rename misspelled column names, and simplify column names. The following columns will be renamed:
average_montly_hours
Work_accident
time_spend_company
Department
```
## Rename columns
df0 = df0.rename(columns={'average_montly_hours': 'average_monthly_hours',
                          'Work_accident': 'work_accident',
                          'time_spend_company': 'tenure',
                          'Department': 'department'})
## Check if columns are renamed
df0
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_monthly_hours</th>
      <th>tenure</th>
      <th>work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>department</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14994</th>
      <td>0.40</td>
      <td>0.57</td>
      <td>2</td>
      <td>151</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
    </tr>
    <tr>
      <th>14995</th>
      <td>0.37</td>
      <td>0.48</td>
      <td>2</td>
      <td>160</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
    </tr>
    <tr>
      <th>14996</th>
      <td>0.37</td>
      <td>0.53</td>
      <td>2</td>
      <td>143</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
    </tr>
    <tr>
      <th>14997</th>
      <td>0.11</td>
      <td>0.96</td>
      <td>6</td>
      <td>280</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
    </tr>
    <tr>
      <th>14998</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>158</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
<p>14999 rows × 10 columns</p>
</div>

#### Check Duplicate Data
```
## Check data for duplicates
print(df0.duplicated().sum())
3008
## Inspect rows containing duplicates
df0[df0.duplicated()].head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_monthly_hours</th>
      <th>tenure</th>
      <th>work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>department</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>396</th>
      <td>0.46</td>
      <td>0.57</td>
      <td>2</td>
      <td>139</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>866</th>
      <td>0.41</td>
      <td>0.46</td>
      <td>2</td>
      <td>128</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1317</th>
      <td>0.37</td>
      <td>0.51</td>
      <td>2</td>
      <td>127</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>1368</th>
      <td>0.41</td>
      <td>0.52</td>
      <td>2</td>
      <td>132</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>RandD</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1461</th>
      <td>0.42</td>
      <td>0.53</td>
      <td>2</td>
      <td>142</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>

The dataset contains 3008 duplicated entries. Upon inspecting the rows containing duplicates, it is unlikely that two employees reporting the same responses for every columns. However, dropping these duplicates could cause imbalance to the dataset, so we'll check the number of duplicated entries for each values of the target variable left.
```
## Create a dataset containing duplicated data
df_duplicates = df0[df0.duplicated()==True]

##Check the number of duplicated values for each value of the target variable
print(df_duplicates['left'].value_counts())
```
```
left
1    1580
0    1428
Name: count, dtype: int64
```
Both of the values for the target variable left have duplicated entries. However, the legitimacy of duplicated entries is questionable and might affect results of the analysis, therefore we will proceed by dropping them.
```
#Drop duplicated rows and save it to a new dataframe
df1 = df0.drop_duplicates(keep = 'first')

df1.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_monthly_hours</th>
      <th>tenure</th>
      <th>work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>department</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>

```
print(df1.shape)
```
(11991, 10)
The dataset is left with 11,991 rows.

#### Check Dataset for Outliers
The tenure column have a max value of more than 2.5x standard deviation. Using a boxplot, we will confirm the outliers.
```
plt.figure(figsize=(3.5,3.5))
plt.title('Boxplot to inspect outliers for `tenure` column',fontweight='bold',fontsize=9)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
sns.boxplot(x=df1['tenure'])
plt.show()
```
![boxplot outlier](https://github.com/ryanlacsamana/Sailfort-Motors-Capstone-Project/assets/138304188/31a6ff31-d580-4621-a464-924f313b0ea0)

The boxplot confirms the presence of outliers in the tenure column. We will now investigate the number of rows that contain outliers in the tenure column.
```
# 25th percentile value for the 'tenure' column
percentile25 = df1['tenure'].quantile(0.25)

# 75th percentile value for the 'tenure' column
percentile75 = df1['tenure'].quantile(0.75)

# Interquartile range in 'tenure' column
iqr = percentile75 - percentile25

# Upper and lower limit for non-outlier values in 'tenure' column
upper_limit = percentile75 + (1.5*iqr)
print('Upper Limit:',upper_limit)
lower_limit = percentile25 - (1.5*iqr)
print('Lower Limit:',lower_limit)

# Subset of data containing outliers in 'tenure' column
outliers = df1[(df1['tenure'] > upper_limit) | (df1['tenure'] < lower_limit)]
print('Number of rows containing outliers in `tenure` column:', len(outliers))
```
```
Upper Limit: 5.5
Lower Limit: 1.5
Number of rows containing outliers in `tenure` column: 824
```
There are 824 rows containing outliers in the tenure column. However, since we will be using tree-based model, which is less sensitive to outliers, we will not remove the outliers.

#### Determine the percentage of employees left to the total number of employees.
```
print(df1['left'].value_counts())
print()
```
```
print(df1['left'].value_counts(normalize=True))
```
```
left
0    10000
1     1991
Name: count, dtype: int64
left
0    0.833959
1    0.166041
Name: proportion, dtype: float64
```
Based on the HR dataset that is free from duplicates, the percentage of employees left to the total number of employees is 16.60%.

### Data Visualization
After data exploration, we will create different plots to investigate the relationship between variables in the HR dataset.

#### a. Relationship between average_monthly_hours and number_project
For this relationship, we will use a box plots and stacked histogram. The boxplot is useful to visualizing the distribution of data, however it does not accurately display the distribution of the data. For the distribution of data, we will use a stacked histogram.
```
fig, ax = plt.subplots(1,2, figsize=(14,7))

## Create boxplot
sns.boxplot(data=df1, x='average_monthly_hours', y='number_project', hue='left', orient='h', ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title("Monthly Hours by Number of Projects", fontweight='bold', fontsize=10)
ax[0].set_xticks([100,120,140,160,180,200,220,240,260,280,300,320])
ax[0].set_xticklabels(ax[0].get_xticks(), rotation=45, fontsize=10)
ax[0].legend(title='turnover')
legend_label = ['stayed','left']
n=0
for i in legend_label:
    ax[0].legend_.texts[n].set_text(i)
    n += 1

## Create histogram
sns.histplot(data=df1, x='number_project', hue='left', multiple='dodge', shrink=2, ax=ax[1])
ax[1].set_title("Histogram for Number of Projects", fontweight='bold')
ax[1].legend(title='turnover', labels=['left','stayed'])

plt.tight_layout()
plt.show()
```
![hist num proj](https://github.com/ryanlacsamana/Sailfort-Motors-Capstone-Project/assets/138304188/d9c08bb9-2a79-4f80-948b-fd596fb6d07d)
The plot shows that the mean number of hours worked for employees who left and stayed increases as the number of project increases. This is expected, as those who worked with more projects tend to work longer hours. However, several things seems to be noticeable in the plots:

There is a noticeable difference in the mean work hours for employees who left versus employees who stayed. The mean number of work hours for employees who left is noticeably lower than those who stayed in the company, but worked on the same number of projects. These could be the employees who underperformed and fired by the company.

For employees who worked with four or more projects, it is also shown that employees who left the company worked more hours that employees who stayed. It also shows that all employees who worked for seven projects left the company. These employees contribute a lot to the company, but this could also be a sign of being overworked.

Assuming a 5-day work week, each day having 8 hours of work, and each employee having a 2-week vacation leave, the number of working hours for this setting is (8 hours/day * 5 days/week * 50 weeks/year) / 12 months = 166.67 hours/month. It seems like majority of employees who work for four or more projects also work more than 166.67 hours/month. It is also evident that majority of these employees left the company. Being overworked could play a major role here.

The ratio of employees who left/stayed who worked for 3 projects is very small. This could indicate that this is the optimal number of projects for employees.

#### b. Relationship between average_monthly_hours and satisfaction_level
Both the average_monthly_hours and satisfaction_level contains quantitative data and continuous variables. For the relationship between the two, we will use scatterplot to clearly represent this relationship.
```
plt.figure(figsize=(9,5))
sns.scatterplot(data=df1, x='average_monthly_hours', y='satisfaction_level', hue='left', alpha=0.4, s=10)
plt.axvline(x=166.67, color='red', linestyle='--')
plt.legend(labels=['166.67 hours/month', 'stayed','left'], loc='upper right', fontsize=7, markerscale=0.5)
plt.title('Relationship Between Average Monthly Hours Worked and Satisfaction Level', fontweight='bold')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.show()
```
![scat hours](https://github.com/ryanlacsamana/Sailfort-Motors-Capstone-Project/assets/138304188/baaf1813-af53-41b8-8303-d6d96ad06e3e)

From the scatterplot, it clearly that employees who worked for more than 166.67 hours/month returned satisfaction level that is close to 0. There are also a sizable group of employees who returned a satisfaction level of around 0.4 but still left the company. However, the distribution of points in the scatterplot seems to be concentrated on certain values, which could indicate possible data manipulation or synthetic data.

#### c. Relationship between tenure and satisfaction_level
For this relationship, we will use boxplot and histogram.
```
fig, ax = plt.subplots(1,2, figsize=(16,7))

# Create boxplot
sns.boxplot(data=df1, x='satisfaction_level', y='tenure', hue='left', orient='h', ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Satisfaction Level by Tenure', fontweight='bold')
ax[0].legend(title='turnover', loc='upper left', fontsize=8)
legend_label = ['stayed','left']
n=0
for i in legend_label:
    ax[0].legend_.texts[n].set_text(i)
    n += 1

# Create histogram
sns.histplot(data=df1, x='tenure', hue='left', multiple='dodge', shrink=5, ax=ax[1])
ax[1].set_title('Tenure Histogram', fontweight='bold')
ax[1].legend(title='turnover', labels=['left','stayed'])
ax[1].set_xticks([2,3,4,5,6,7,8,9,10])

plt.show()
```
![hist tenure](https://github.com/ryanlacsamana/Sailfort-Motors-Capstone-Project/assets/138304188/ce494860-c13b-4de6-bdf6-eb5593e41d20)

The following observations are obtained from the plots:

Employees with the longest tenures (more than 6 years) stayed.
Employees with the longest tenures also have high satisfaction levels for the company. The satisfaction levels from these employees also aligned to new employees and employees with shorter tenures.
Employees who left the company after 4 years of stay yield a very low satisfaction level.
Employees with medium-length tenures (5 to 6 years) gave high satisfaction rating but still left the company.
There are few longer-tenured employees. This could be employees who are high-ranking or higher-paid.
Next, we will calculate the mean and median satisfaction levels for employees who left and those who stayed.
```
df1.groupby(['left'])['satisfaction_level'].agg(['mean','median'])
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>left</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.667365</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.440271</td>
      <td>0.41</td>
    </tr>
  </tbody>
</table>
</div>
