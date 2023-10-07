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

Employees who left the company gave a mean and median satisfaction level of 0.44 and 0.41, respectively. These values are lower than the mean and median satisfaction level from employees who stayed in the company. Also, mean satisfaction score from employees who stayed is quite lower than the median, which could indicate that the satisfaction level for employees who stayed is skewed to the left.

#### d. Salary Levels for Different Tenures
```
fig, ax = plt.subplots(1,2, figsize=(14,7))

## Define tenure classification
## Long-tenured employees (more than 6 years)
## Short-tenured employees (6 years and less)
tenure_long = df1[df1['tenure'] > 6]
tenure_short = df1[df1['tenure'] < 7]

## Plot Long-tenured histogram
sns.histplot(data=tenure_long, x='tenure', hue='salary', discrete=1, hue_order=['low','medium','high'], multiple='dodge', shrink=0.5, ax=ax[0])
ax[0].set_title('Salary Histogram for Long-tenured Employees', fontweight='bold')

## Plot Short-tenured histogram
sns.histplot(data=tenure_short, x='tenure', hue='salary', discrete=1, hue_order=['low','medium','high'], multiple='dodge', shrink=0.5, ax=ax[1])
ax[1].set_title('Salary Histogram for Short-tenured Employees', fontweight='bold')


plt.show()
```
![salary hist](https://github.com/ryanlacsamana/Sailfort-Motors-Capstone-Project/assets/138304188/272a0f65-e87f-4039-a395-73678e83a38e)

The plot above shows that majority of long-tenured employees have medium salaries, however there is a little discrepancy between the number of low, medium, and highly-paid long-tenured employees, unlike for short-tenured employees, which has a very large number of low and medium-paid employees compared to highly-paid employees.

#### e. Relationship between last_evaluation and average_monthly_hours
We will create a scatterplot to determine if there is a correlation between working long hours and receiving high evaluation points.
```
plt.figure(figsize=(9,5))
sns.scatterplot(data=df1, x='average_monthly_hours', y='last_evaluation', hue='left', alpha=0.4, s=10)
plt.axvline(x=166.67, color='red', linestyle='--')
plt.legend(labels=['166.67 hours/month','stayed','left'], fontsize=7)
plt.title('Relationship between Monthly Hours Worked and Evaluation Score', fontweight='bold')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.show()
```
![scat work eval](https://github.com/ryanlacsamana/Sailfort-Motors-Capstone-Project/assets/138304188/c4b8cec7-b992-4bba-bae5-5c8075393c5f)

The relationship yields the following observations:

1.  There are two types of employees who left the company. The first group are employees who worked more than the monthly average of 166.67 hours and received high evaluation scores, these could indicate employees with high productivity but overworked. The second group are employees who worked less than the monthly average and received low evaluation scores.
2.  It seems like a noticeable amount of employees in the far upper right quadrant worked the longest hours and left the company, despite having high evaluation scores.
3.  There seems to be a correlation between the number of hours worked and evaluation scores.
4.  Despite having a correlation, working more than the monthly average does not guarantee high evaluation scores.
5.  Based on the distribution of points in the scatterplot, most of the employees worked more than the monthly average of 166.67 hours/month.

#### f. Relationship between average_monthly_hours and promotion_last_5years
Inspect whether employees who worked more than the monthly average were promoted in the last 5 years.
```
figsize=(7,7)

sns.scatterplot(data=df1, x='average_monthly_hours', y='promotion_last_5years', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='red', linestyle='--')
plt.title('Monthly Hours by Promotion last 5 Years', fontweight='bold')
plt.legend(labels=['166.67 hours/month','stayed','left'], fontsize=7)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.show()
```
![scat prom](https://github.com/ryanlacsamana/Sailfort-Motors-Capstone-Project/assets/138304188/5e8ee729-64ac-4c27-bfed-6a53da98b57c)

The plot above shows the following observations:
1.  There are very few employees who worked more than the monthly average were promoted.
2.  The employees who worked the longest hours, and left the company, were not promoted in the last 5 years.
3.  Majority of employees who left worked far more than the monthly average.
   
#### g. Distribution of Employees between Departments
Inspect the distribution of employees between departments:
```
print(df1['department'].value_counts())
```
```
department
sales          3239
technical      2244
support        1821
IT              976
RandD           694
product_mng     686
marketing       673
accounting      621
hr              601
management      436
Name: count, dtype: int64
```
Plot a histogram to compare the distribution of employees of left and stayed for each department.
```
plt.subplots(figsize=(9,7))

sns.histplot(data=df1, x='department', hue='left', multiple='dodge', shrink=0.4)
plt.legend(labels=['left','stayed'])
plt.xticks(rotation=45, fontsize=10)
plt.title('Count of Employees Stayed/Left per Department', fontweight='bold')

plt.show()
```
![count stay left](https://github.com/ryanlacsamana/Sailfort-Motors-Capstone-Project/assets/138304188/45e3da36-6a37-4343-a4f3-bb83c476363d)

Based on the histogram above, all departments have a higher number of employees who stayed versus employees who left.

#### h. Correlations between Variables in the Data
To clearly visualize the relationship between each variables in the dataset, we will use a heatmap.
```
heatmap_corr=df1[['satisfaction_level','last_evaluation','number_project','average_monthly_hours','tenure','work_accident','left','promotion_last_5years']].corr()

plt.figure(figsize=(8,5))

heatmap = sns.heatmap(heatmap_corr, annot=True, annot_kws={'fontsize': 6}, vmin=-1, vmax=1, cmap=sns.diverging_palette(220,20, as_cmap=True))
heatmap.set_title('Correlation Heatmap', fontweight='bold', fontsize=12, pad=12)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.show()
```
![corr heatmap](https://github.com/ryanlacsamana/Sailfort-Motors-Capstone-Project/assets/138304188/c1891813-6ded-41a5-ba74-b248b1a4389c)

The correlation heatmap shows positive correlation for the following values - average_monthly_hours, number_project, and last_evaluation, while showing negative correlation for left, and satisfaction_level.

#### i. Insights
Based on the relationship between variables, it could imply that employees are leaving due to long working hours and number of projects, it seems that these employees are burned out, which could reflect in the employee's satisfaction level.
Employees who left did not receive promotion in the last 5 years, despite having high evaluation scores.
An employee who spends more than 6 years in the company tends not to leave, however, there are very few employees who reached this tenure.

### Model Building, Results and Evaluation
The goal for this project is to predict whether an employee will leave the company. The outcome is a categorical variable and involves binary classification. The outcome variable will be 1 (employee left the company) and 0 (employee did not leave the company).

#### Modeling Approach A: Logistic Regression Model
The dataset includes 2 non-numeric variables, department and salary. The variable department is a categorical variable and can be converted as a dummy in modeling. The variable salary is ordinal, and can be converted into numbers 0-2.
```
## Copy the dataframe
df_enc = df1.copy()

## Encode the 'salary' column as an ordinal numeric category
df_enc['salary'] = (df_enc['salary'].astype('category').cat.set_categories(['low','medium','high']).cat.codes)

## Dummy encode the 'department' column
df_enc = pd.get_dummies(df_enc, drop_first=False)

## Display the new dataframe
df_enc.head()
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
      <th>salary</th>
      <th>department_IT</th>
      <th>department_RandD</th>
      <th>department_accounting</th>
      <th>department_hr</th>
      <th>department_management</th>
      <th>department_marketing</th>
      <th>department_product_mng</th>
      <th>department_sales</th>
      <th>department_support</th>
      <th>department_technical</th>
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
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
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
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
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
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
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
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
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
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>

```
## Create a heatmap to visualize the correlation between variables
heatmap_corr_df_enc = df_enc[['satisfaction_level','last_evaluation','number_project','average_monthly_hours','tenure']].corr()

plt.figure(figsize=(6,4))

sns.heatmap(heatmap_corr_df_enc, annot=True, annot_kws={'fontsize': 7}, cmap=sns.color_palette('vlag', as_cmap=True))
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('Heatmap of the Dataset', fontweight='bold')

plt.show()
```

![heatmap logreg](https://github.com/ryanlacsamana/Sailfort-Motors-Capstone-Project/assets/138304188/0695e0ff-30cd-4c54-a65a-4378cd2f99eb)

```
## Create a plot to visualize the number of employees who stayed/left per department
plt.subplots(figsize=(7,5))

sns.histplot(data=df1, x='department', hue='left', multiple='dodge', shrink=0.4)
plt.legend(labels=['left','stayed'], fontsize=8)
plt.xticks(rotation=45, fontsize=9)
plt.title('Count of Employees Stayed/Left per Department', fontweight='bold')

plt.show()
```

![emp left stay](https://github.com/ryanlacsamana/Sailfort-Motors-Capstone-Project/assets/138304188/ba0a3824-ded8-4b04-9b96-92ca092b6699)

Logistic regression is sensitive to outliers. It is best to remove them before proceeding. We will the outliers from the 'tenure' column that were identified earlier.
```
## Select rows without outliers in the 'tenure' column and include them in a new dataset
df_logreg = df_enc[(df_enc['tenure'] >= lower_limit) & (df_enc['tenure'] <= upper_limit)]

df_logreg.head()
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
      <th>salary</th>
      <th>department_IT</th>
      <th>department_RandD</th>
      <th>department_accounting</th>
      <th>department_hr</th>
      <th>department_management</th>
      <th>department_marketing</th>
      <th>department_product_mng</th>
      <th>department_sales</th>
      <th>department_support</th>
      <th>department_technical</th>
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
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
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
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
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
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
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
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
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
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>

#### Isolate the Outcome Variable
```
y = df_logreg['left']

print(y.head())
```
```
0    1
2    1
3    1
4    1
5    1
Name: left, dtype: int64
```

#### Select features to be used in the model
```
X = df_logreg.drop('left', axis=1)

X.head()
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
      <th>promotion_last_5years</th>
      <th>salary</th>
      <th>department_IT</th>
      <th>department_RandD</th>
      <th>department_accounting</th>
      <th>department_hr</th>
      <th>department_management</th>
      <th>department_marketing</th>
      <th>department_product_mng</th>
      <th>department_sales</th>
      <th>department_support</th>
      <th>department_technical</th>
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
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.41</td>
      <td>0.50</td>
      <td>2</td>
      <td>153</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>

#### Split the data into training and testing sets
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
```
#### Construct a Logistic Regression Model and fit it to the training dataset

```
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)
```
#### Use the Logistic Regression Model to make predictions of the test set
```
y_pred = log_clf.predict(X_test)
```

#### Create a Confusion Matrix
```
## Compute values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)

## Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=['stayed','left'])

## Plot confusion matrix
log_disp.plot(values_format='', cmap='YlOrBr')

plt.show()
```

![confusionmatrix logreg](https://github.com/ryanlacsamana/Sailfort-Motors-Capstone-Project/assets/138304188/370195bc-ee83-4e28-87e7-ff7e82049677)

The confusion matrix returned the following values:

* **True negatives** (employees who did not leave the company that the model accurately predicted did not leave)
* **False negatives** (employees who did not leave the company but the model predicted as leaving)
* **True positives** (employees who left the company that the model accurately predicted as leaving)
* **False positives** (employees who left the company but the model predicted did not leave)

#### Create a classification report
```
## Check the class balance in the data
print(df_logreg['left'].value_counts(normalize=True))
left
0    0.831468
1    0.168532
Name: proportion, dtype: float64
```
The data is composed of approximately 83% employees who stayed and approximately 17% employees who left the company. The data is not perfectly balanced, but is not too imbalanced. We can continue evaluating the model without modifying the class balance.
```
## Create a classification report for logistic regression model
target_names = ['Predicted would not leave','Predicted would leave']
print(classification_report(y_test, y_pred, target_names=target_names))
```
```
                           precision    recall  f1-score   support

Predicted would not leave       0.86      0.93      0.90      2321
    Predicted would leave       0.44      0.26      0.33       471

                 accuracy                           0.82      2792
                macro avg       0.65      0.60      0.61      2792
             weighted avg       0.79      0.82      0.80      2792
```

The model values of precision, recall, and f1-score of 0.44, 0.26, and 0.33 respectively. These scores are significantly low. However, it is important to note that we removed outliers in the dataset. These outliers might have been significant data points to be considered in the model. With this, we will consider Tree-based modeling.

### Modeling Approach B: Tree-based Model
#### Isolate the outcome variable
```
y = df_enc['left']

print(y.head())
```
```
0    1
1    1
2    1
3    1
4    1
Name: left, dtype: int64
```
#### Select the features
```
X = df_enc.drop('left', axis=1)

X.head()
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
      <th>promotion_last_5years</th>
      <th>salary</th>
      <th>department_IT</th>
      <th>department_RandD</th>
      <th>department_accounting</th>
      <th>department_hr</th>
      <th>department_management</th>
      <th>department_marketing</th>
      <th>department_product_mng</th>
      <th>department_sales</th>
      <th>department_support</th>
      <th>department_technical</th>
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
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>

#### Split the data into training and testing sets
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```
#### Round 1

#### Decision Tree
```
## Instantiate model
tree = DecisionTreeClassifier(random_state=0)

## Assign a dictionary of hyperparameters
cv_params = {'max_depth': [4,6,8,None], 'min_samples_leaf': [2,5,1], 'min_samples_split': [2,4,6]}

## Assign scoring metrics to capture
scoring = ('accuracy','precision','recall','f1','roc_auc')

## Instantiate Grid Search
tree1 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')
```

#### - Fit the decision tree model to the training data
```
%%time
tree1.fit(X_train, y_train)
CPU times: user 3.6 s, sys: 3.1 ms, total: 3.61 s
Wall time: 3.61 s
```
```
GridSearchCV(cv=4, estimator=DecisionTreeClassifier(random_state=0),
             param_grid={'max_depth': [4, 6, 8, None],
                         'min_samples_leaf': [2, 5, 1],
                         'min_samples_split': [2, 4, 6]},
             refit='roc_auc',
             scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'))
estimator: DecisionTreeClassifier
DecisionTreeClassifier(random_state=0)

DecisionTreeClassifier
DecisionTreeClassifier(random_state=0)
```

#### - Identify optimal values for the decision tree parameters
```
print("Best parameters:", tree1.best_params_)
print("Best AUC score for CV:", tree1.best_score_)
Best parameters: {'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 4}
Best AUC score for CV: 0.9688735287591919
```
The model returned an AUC score of 0.969, which is pretty high. This indicates how well the model can predict employees who will leave.

#### **- Write a function to extract all the scores from the grid search**
```
def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, accuracy, or auc
    
    Returns a pandas df with the f1, recall, precision, accuracy, and auc scores for the model with the best mean 'metric' score across validation folds.
    '''

    ## Dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'auc': 'mean_test_roc_auc','precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy'
               }

    ## Get all the results from the CV and put them in a dataframe
    cv_results =  pd.DataFrame(model_object.cv_results_)

    ## Isolate the row of the dataframe with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    ## Extract accuracy, precision, recall, and f1 score from that row
    auc = best_estimator_results.mean_test_roc_auc
    precision = best_estimator_results.mean_test_precision
    recall = best_estimator_results.mean_test_recall
    f1 = best_estimator_results.mean_test_f1
    accuracy = best_estimator_results.mean_test_accuracy

    ## Create table of results
    table = pd.DataFrame()
    table = pd.DataFrame({'model':[model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'f1': [f1],
                          'accuracy': [accuracy],
                          'auc': [auc]
                      })
    return table
```
Get the scores from the GridSearch
```
## Get all CV scores
tree1_cv_results = make_results('Decision Tree CV 1', tree1, 'auc')
tree1_cv_results
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
      <th>accuracy</th>
      <th>auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Decision Tree CV 1</td>
      <td>0.963198</td>
      <td>0.922352</td>
      <td>0.942266</td>
      <td>0.981208</td>
      <td>0.968874</td>
    </tr>
  </tbody>
</table>
</div>

The model returns high scores for precision, recall, f1, accuracy, and auc, which is strong indicator of the model's good performance. Decision trees can be vulnerable to overfitting, however, random forest can avoid overfitting by incorporating multiple trees to make predictions.

#### Random Forest

#### - Construct a random forest model and set-up cross-validated grid search

#### -Fit the random forest model to the training data
```
## Instantiate model
rf = RandomForestClassifier(random_state=0)

## Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[3,5,None],
             'max_features':[1.0],
             'max_samples':[0.7,1.0],
             'min_samples_leaf':[1,2,3],
             'min_samples_split':[2,3,4],
             'n_estimators':[300,500]
             }

## Assign scoring metrics to capture
scoring = ('accuracy','precision','recall','f1','roc_auc')

## Instantiate GridSearch
rf1 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')
```
#### - Fit the random forest model to the training data
```
%%time
rf1.fit(X_train, y_train)
CPU times: user 22min 4s, sys: 2.74 s, total: 22min 7s
Wall time: 22min 10s
```
```
GridSearchCV
GridSearchCV(cv=4, estimator=RandomForestClassifier(random_state=0),
             param_grid={'max_depth': [3, 5, None], 'max_features': [1.0],
                         'max_samples': [0.7, 1.0],
                         'min_samples_leaf': [1, 2, 3],
                         'min_samples_split': [2, 3, 4],
                         'n_estimators': [300, 500]},
             refit='roc_auc',
             scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'))
estimator: RandomForestClassifier
RandomForestClassifier(random_state=0)

RandomForestClassifier
RandomForestClassifier(random_state=0)
```

Write pickle to save the model
```
## Write pickle
pd.to_pickle(rf1, "sailfort-motors", compression='infer', protocol=5)
## Read pickle
pd.read_pickle("sailfort-motors", compression='infer')
GridSearchCV
GridSearchCV(cv=4, estimator=RandomForestClassifier(random_state=0),
             param_grid={'max_depth': [3, 5, None], 'max_features': [1.0],
                         'max_samples': [0.7, 1.0],
                         'min_samples_leaf': [1, 2, 3],
                         'min_samples_split': [2, 3, 4],
                         'n_estimators': [300, 500]},
             refit='roc_auc',
             scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'))
estimator: RandomForestClassifier
RandomForestClassifier(random_state=0)

RandomForestClassifier
RandomForestClassifier(random_state=0)
```
#### -Identify the best AUC score and optimal values for the parameters by the random forest model
```
## Check best AUC score
print("Best AUC Score from the RF model:")
rf1.best_score_
```
Best AUC Score from the RF model:
0.9790751171477935
```
## Check best parameters
print("Optimal values for the parameters of the RF model:")
rf1.best_params_
```
```
Optimal values for the parameters of the RF model:
{'max_depth': None,
 'max_features': 1.0,
 'max_samples': 0.7,
 'min_samples_leaf': 2,
 'min_samples_split': 2,
 'n_estimators': 500}
```

#### - Collect all the evaluation scores on the training set for the decision tree and random forest models
```
rf1_cv_results = make_results('Random Forest CV 1', rf1, 'auc')
tree_cv_results = pd.concat([tree1_cv_results, rf1_cv_results], axis=0)
tree_cv_results = tree_cv_results.reset_index(drop=True)
tree_cv_results
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
      <th>accuracy</th>
      <th>auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Decision Tree CV 1</td>
      <td>0.963198</td>
      <td>0.922352</td>
      <td>0.942266</td>
      <td>0.981208</td>
      <td>0.968874</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Random Forest CV 1</td>
      <td>0.985667</td>
      <td>0.916999</td>
      <td>0.950078</td>
      <td>0.983988</td>
      <td>0.979075</td>
    </tr>
  </tbody>
</table>
</div>

The random forest model returned evaluation scores which are higher than the scores from the decision tree model, with the exception for recall. The recall score from the random forest model is lower by 0.005353 than the score from the decision tree model, however, this is negligible. Overall, the random forest has better performance compared to the decision tree model.

#### - Evaluate the final model on the test set
```
## Define a function that gets all the scores from the model's predictions
def get_scores(model_name:str, model, X_test_data, y_test_data):
    '''
    Generate a table of tests scores.
    
    In:
        model_name(string): How you want your model to be named in the output table
        model: A fit GridSearchCV object
        X_test_data: numpy array of X_test data
        y_test_data: numpy array of y_test data
        
    Out:
        pandas dataframe of precision, recall, f1, accuracy, and AUC scores for your model
    '''
    
    preds = model.best_estimator_.predict(X_test_data)
    
    auc = roc_auc_score(y_test_data, preds)
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)
    
    table = pd.DataFrame({'model':[model_name],
                          'precision':[precision],
                          'recall':[recall],
                          'f1':[f1],
                          'accuracy':[accuracy],
                          'AUC':[auc]
                          })
    
    return table
```
```
## Get predictions on test data
rf1_test_scores = get_scores('Random Forest 1 Test Scores', rf1, X_test, y_test)
rf1_test_scores
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
      <th>accuracy</th>
      <th>AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Random Forest 1 Test Scores</td>
      <td>0.984716</td>
      <td>0.907445</td>
      <td>0.944503</td>
      <td>0.982322</td>
      <td>0.952323</td>
    </tr>
  </tbody>
</table>
</div>    

The test scores are almost similar to the validation scores, which means that this is a strong model. This model will perform well on new and unseen data.

#### Feature Engineering

The model yields high evaluation scores. However, there is a chance that a data leakage occured.

It might be possible that the satisfaction level for all the employees were not reported. There could be also be employees that worked few hours because they decided to quit the job, or the management decided to fire them.

The previous round of decision tree and random forest models included all variables as features. For this round, we will drop the satisfaction_level and average_monthly_hours and create a new feature overworked, which contains a binary variable on whether an employee is overworked.
```
## Drop 'satisfaction_level' and save the resulting dataframe into a new variable
df2 = df_enc.drop('satisfaction_level', axis=1)

df2.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_monthly_hours</th>
      <th>tenure</th>
      <th>work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>salary</th>
      <th>department_IT</th>
      <th>department_RandD</th>
      <th>department_accounting</th>
      <th>department_hr</th>
      <th>department_management</th>
      <th>department_marketing</th>
      <th>department_product_mng</th>
      <th>department_sales</th>
      <th>department_support</th>
      <th>department_technical</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>

```
## Create 'overworked' column which is identical to the 'average_monthly_hours' column
df2['overworked'] = df2['average_monthly_hours']

## Check the max and min values for 'overworked' column
print("Max hours:", df2['overworked'].max())
print("Min hours:", df2['overworked'].min())
Max hours: 310
Min hours: 96
```
For our assumption of a 5-days work week, 8 hours per day, with 2 weeks of vacation each year, the average monthly working hours will be 166.67 hours. Let us account for an overtime of 8 hours or weekend work of 8 hours per week, the threshold for being overworked will be **(8 hours/day * 6 days/week * 50 weeks/year) / 12 months/year = 200 hours**

If df['overworked'] > 200, it will return True, and if df['overworked'] ≤ 200, it will return False. The results will be converted to int, which is 1 for True and 0 for False.
```
## Create 'overworked' column
df2['overworked'] = (df2['overworked'] > 200).astype(int)

## Drop the 'average_monthly_hours' column
df2 = df2.drop('average_monthly_hours', axis=1)


df2.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>tenure</th>
      <th>work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>salary</th>
      <th>department_IT</th>
      <th>department_RandD</th>
      <th>department_accounting</th>
      <th>department_hr</th>
      <th>department_management</th>
      <th>department_marketing</th>
      <th>department_product_mng</th>
      <th>department_sales</th>
      <th>department_support</th>
      <th>department_technical</th>
      <th>overworked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.53</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.86</td>
      <td>5</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.88</td>
      <td>7</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.87</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.52</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

#### Round 2

#### Decision Tree

#### - Isolate features and target variable
```
y = df2['left']

X = df2.drop('left', axis=1)
```
#### - Create test data
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
```
#### - Instantiate model
```
## Instantiate model
tree = DecisionTreeClassifier(random_state=0)

## Assign a dictionary of hyperparameters
cv_params = {'max_depth':[4,6,8,None],
             'min_samples_leaf':[2,5,1],
             'min_samples_split':[2,4,6]
             }

## Assign scoring metrics
scoring = ('accuracy','precision','recall','f1','roc_auc')

## Instantiate GridSearch
tree2 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')
```
```
%%time
tree2.fit(X_train, y_train)
CPU times: user 3.04 s, sys: 3 ms, total: 3.05 s
Wall time: 3.05 s
```
```
GridSearchCV
GridSearchCV(cv=4, estimator=DecisionTreeClassifier(random_state=0),
             param_grid={'max_depth': [4, 6, 8, None],
                         'min_samples_leaf': [2, 5, 1],
                         'min_samples_split': [2, 4, 6]},
             refit='roc_auc',
             scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'))
estimator: DecisionTreeClassifier
DecisionTreeClassifier(random_state=0)

DecisionTreeClassifier
DecisionTreeClassifier(random_state=0)
```
#### - Check the best parameter and AUC scores
```
## Check best parameters
print("Best parameters:", tree2.best_params_)

## Check best AUC score
print("Best AUC score:", tree2.best_score_)
```
Best parameters: {'max_depth': 6, 'min_samples_leaf': 5, 'min_samples_split': 2}
Best AUC score: 0.956906013294911

#### - Check other scores
```
## Get all CV results
tree2_cv_results = make_results('Decision Tree CV 2', tree2, 'auc')

tree_cv_results2 = pd.concat([tree1_cv_results, tree2_cv_results], axis=0)
tree_cv_results2 = tree_cv_results2.reset_index(drop=True)
tree_cv_results2
```
The model still yields high AUC score even after dropping the satisfaction level and average monthly hours column.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
      <th>accuracy</th>
      <th>auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Decision Tree CV 1</td>
      <td>0.963198</td>
      <td>0.922352</td>
      <td>0.942266</td>
      <td>0.981208</td>
      <td>0.968874</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Decision Tree CV 2</td>
      <td>0.875433</td>
      <td>0.900874</td>
      <td>0.887608</td>
      <td>0.962082</td>
      <td>0.956906</td>
    </tr>
  </tbody>
</table>
</div>

The model yields slightly lower scores on all metrics, except for precision. However, this is still indicative of a strong model.

#### Random Forest
```
## Instantiate model
rf = RandomForestClassifier(random_state=0)

## Assign a dictionary of hyperparamaters
cv_params = {'max_depth':[3,5,None],
             'max_features':[1.0],
             'max_samples':[0.7,1.0],
             'min_samples_leaf':[1,2,3],
             'min_samples_split':[2,3,4],
             'n_estimators':[300,500]
             }

## Assign scoring metrics
scoring = ('accuracy','recall','precision','f1','roc_auc')

## Instantiate GridSearch
rf2 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')
```
```
%%time
rf2.fit(X_train, y_train)
CPU times: user 16min 24s, sys: 2.62 s, total: 16min 26s
Wall time: 16min 28s
```
```
GridSearchCV
GridSearchCV(cv=4, estimator=RandomForestClassifier(random_state=0),
             param_grid={'max_depth': [3, 5, None], 'max_features': [1.0],
                         'max_samples': [0.7, 1.0],
                         'min_samples_leaf': [1, 2, 3],
                         'min_samples_split': [2, 3, 4],
                         'n_estimators': [300, 500]},
             refit='roc_auc',
             scoring=('accuracy', 'recall', 'precision', 'f1', 'roc_auc'))
estimator: RandomForestClassifier
RandomForestClassifier(random_state=0)

RandomForestClassifier
RandomForestClassifier(random_state=0)
```
Write pickle to save the model
```
pd.to_pickle(rf2, "sailfort-motors", compression='infer', protocol=5)
pd.read_pickle("sailfort-motors", compression='infer')
```
```
GridSearchCV
GridSearchCV(cv=4, estimator=RandomForestClassifier(random_state=0),
             param_grid={'max_depth': [3, 5, None], 'max_features': [1.0],
                         'max_samples': [0.7, 1.0],
                         'min_samples_leaf': [1, 2, 3],
                         'min_samples_split': [2, 3, 4],
                         'n_estimators': [300, 500]},
             refit='roc_auc',
             scoring=('accuracy', 'recall', 'precision', 'f1', 'roc_auc'))
estimator: RandomForestClassifier
RandomForestClassifier(random_state=0)

RandomForestClassifier
RandomForestClassifier(random_state=0)
```
```
#### - Check for optimal parameters, best AUC scores, and get all the CV scores

## Check best parameters
print("Best parameters:", rf2.best_params_)

## Check best AUC score
print("Best AUC score:", rf2.best_score_)
```
Best parameters: {'max_depth': 5, 'max_features': 1.0, 'max_samples': 1.0, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 500}
Best AUC score: 0.9663699459505957
```
## Get all CV scores
rf2_cv_results = make_results('Random Forest CV 2', rf2, 'auc')
tree_rf_cv_results2 = pd.concat([tree1_cv_results, tree2_cv_results, rf1_cv_results, rf2_cv_results], axis=0)
tree_rf_cv_results2 = tree_rf_cv_results2.reset_index(drop=True)
tree_rf_cv_results2
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
      <th>accuracy</th>
      <th>auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Decision Tree CV 1</td>
      <td>0.963198</td>
      <td>0.922352</td>
      <td>0.942266</td>
      <td>0.981208</td>
      <td>0.968874</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Decision Tree CV 2</td>
      <td>0.875433</td>
      <td>0.900874</td>
      <td>0.887608</td>
      <td>0.962082</td>
      <td>0.956906</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Random Forest CV 1</td>
      <td>0.985667</td>
      <td>0.916999</td>
      <td>0.950078</td>
      <td>0.983988</td>
      <td>0.979075</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Random Forest CV 2</td>
      <td>0.862286</td>
      <td>0.904222</td>
      <td>0.882598</td>
      <td>0.960080</td>
      <td>0.966370</td>
    </tr>
  </tbody>
</table>
</div>

Compared to the first model for Decision Trees and Random Forest, respectively, the scores drop for the second model. However, these scores are still high and can perform well.

The Random Forest model scored better on recall and auc metrics, while the Decision Tree model scored better for precision, f1, and accuracy metrics. The Decision tree scored higher on f1 score than the Random Forest model, but by a very small margin, so we will get predictions on test data from both models.

#### - Get predictions on test data
```
rf2_test_scores = get_scores('Random Forest Test 2', rf2, X_test, y_test)

tree2_test_scores = get_scores('Decision Tree Test 2', tree2, X_test, y_test)
test2_prediction_results = pd.concat([tree2_test_scores, rf2_test_scores], axis=0)
test2_prediction_results
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
      <th>accuracy</th>
      <th>AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Decision Tree Test 2</td>
      <td>0.792683</td>
      <td>0.913655</td>
      <td>0.848881</td>
      <td>0.945964</td>
      <td>0.933027</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Random Forest Test 2</td>
      <td>0.882812</td>
      <td>0.907631</td>
      <td>0.895050</td>
      <td>0.964643</td>
      <td>0.941815</td>
    </tr>
  </tbody>
</table>
</div>

Both the test scores also yields high scores and almost similar to the results of the validation, except for the precision score from the Decision Tree model.

#### Plot a confusion matrix to visualize how well the model performs
```
fig, ax = plt.subplots(1,2, figsize=(16,7))

## CONFUSION MATRIX FROM DECISION TREE MODEL
## Generate array of values for confusion matrix
preds = tree2.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=rf2.classes_)

## Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['stayed','left'])
disp.plot(values_format='', cmap='YlOrBr', ax=ax[0])
ax[0].set_title("Confusion Matrix from Decision Tree Model")

## CONFUSION MATRIX FROM RANDOM FOREST MODEL
## Generate array of values for confusion matrix
preds = rf2.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=rf2.classes_)

## Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['stayed','left'])
disp.plot(values_format='', cmap='YlOrBr', ax=ax[1])
ax[1].set_title("Confusion Matrix from Random Forest Model")

plt.show()
```

![confusionmatrix logreg dt vs rf](https://github.com/ryanlacsamana/Sailfort-Motors-Capstone-Project/assets/138304188/e8d25e4f-c4ab-40c9-8acd-55c62b839acb)

#### Various Exploratory Plots

#### Decision Tree Splits
```
## Plot the tree
plt.figure(figsize=(85,20))

plot_tree(tree2.best_estimator_, max_depth=6, fontsize=10, feature_names=X.columns, class_names={0:'stayed',1:'left'}, filled=True)

plt.show()
```

![dt](https://github.com/ryanlacsamana/Sailfort-Motors-Capstone-Project/assets/138304188/51bfdb52-64fd-416f-8b36-30501730d1c0)

#### Decision tree feature importance
```
## tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_, columns=X_columns)

tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_, columns=['gini_importance'], index=X.columns)

tree2_importances = tree2_importances.sort_values(by='gini_importance', ascending=False)
tree2_importances
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gini_importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>last_evaluation</th>
      <td>0.349432</td>
    </tr>
    <tr>
      <th>number_project</th>
      <td>0.299012</td>
    </tr>
    <tr>
      <th>tenure</th>
      <td>0.196538</td>
    </tr>
    <tr>
      <th>overworked</th>
      <td>0.152354</td>
    </tr>
    <tr>
      <th>department_technical</th>
      <td>0.001271</td>
    </tr>
    <tr>
      <th>work_accident</th>
      <td>0.000537</td>
    </tr>
    <tr>
      <th>salary</th>
      <td>0.000440</td>
    </tr>
    <tr>
      <th>department_support</th>
      <td>0.000412</td>
    </tr>
    <tr>
      <th>department_sales</th>
      <td>0.000004</td>
    </tr>
    <tr>
      <th>promotion_last_5years</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>department_IT</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>department_RandD</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>department_hr</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>department_management</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>department_marketing</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>department_product_mng</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>department_accounting</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>

```
## Create a bar plot of decision tree feature importance

sns.barplot(data=tree2_importances, x='gini_importance', y=tree2_importances.index, orient='h')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Decision Tree: Feature Importance of Employee Leaving')
plt.show()
```

![dt feat imp](https://github.com/ryanlacsamana/Sailfort-Motors-Capstone-Project/assets/138304188/9a497a8c-f8cb-45d6-951e-e38ea77c4a17)

#### Random forest feature importance
```
## Get feature importances
rf2_importances = rf2.best_estimator_.feature_importances_

## Get indices of top 10 features
ind = np.argpartition(rf2.best_estimator_.feature_importances_,-10)[-10:]

## Get columns labes of top 10 features
feat = X.columns[ind]

## Filter 'rf2_importances' to consist of top 10 feature importances
rf2_importances = rf2_importances[ind]

y_df = pd.DataFrame({'Feature': feat,
                     'Importance': rf2_importances
                     })
y_sort_df = y_df.sort_values('Importance')
fig = plt.figure()
ax1 = fig.add_subplot(111)
```

![rf feat imp](https://github.com/ryanlacsamana/Sailfort-Motors-Capstone-Project/assets/138304188/009657d1-883e-46c0-aa36-f3bd7099569c)

From the bar plots of Feature Importance from both Decision Tree and Random Forest, the variables last_evaluation, number_project, tenure, and overworked holds the most importance. These are the same variables that we used in developing the decision tree model.

### Conclusions and Recommendations
The models and feature importance confirmed that the employees in the company are overworked.

To increase employee retention, the following recommendations could be presented to the stakeholders:

  1. Employees who worked longer hours and received high evaluation scores should be considered for promotion.
  2. Conduct investigation for the dissatisfaction of employees of employees who reach the tenure of 4 years.
  3. Limit the number of projects assigned for each employee.
  4. Reward employees who worked for longer hours. The type of reward could be based on evaluation scores.
  5. Evaluation scores should not only be based on the number of working hours rendered by an employee. Consider other metrics for evaluating employees.
  6. Familiarize employees regarding overtime pay policies, workloads, and time-offs.
  7. The company should conduct company-wide and team discussions to understand and address issues regarding workloads, work cultures, salary, and other specific and relevant contexts.

To further improve the results of the study, other aspects should be considered that were not be included in the dataset. These aspects include but not limited to the following:

* Frequency of salary increase of each employee: The dataset did not include if an employee received a salary increase for a certain amount of time. This could also be driving an employee to stay or leave the company.
* Distance of employee from the work site: Work-from-home or hybrid setup could be implemented for employees who have work assignments that can be done remotely. This could also save the company from expenses such as electricity. For employees with workloads that cannot be done remotely, but are living far from the work site, consider renting a unit or house to be used by these employees.
* Conduct activities such as seminars, trainings, and recreational activities such as team building, to improve employee skills and relationship with other employees.

