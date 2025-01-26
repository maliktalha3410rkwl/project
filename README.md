## Data read
import pandas as pd
df =pd.read_csv("data.csv")
## data cleaning
info how many column and row
df.info()
### we want to read all column
df = pd.DataFrame(df)
df.head()
#### slicing
df[2:5]
check column to avoid error
column = list(df.columns)
column
### Drop column that isnot needed
df.drop(columns='Row ID',inplace=True)  

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

### Check n/a values
df.isna().sum()
### Dropping n/a values and check again
df  =df.dropna(subset=["Postal Code"])
df.isna().sum()
### Check duplicate
df.duplicated('Order Date').sum()
### Drop duplicate
df.drop_duplicates(subset='Order Date',inplace=True)
df.duplicated('Order Date').sum()
### Again check
df.duplicated('Order Date').sum()
### Description
df.describe()
### If you want to seprate thing by commas
df['Product Name'].str.split(',',expand=True)
## Data Agregation
### Applying aggregation across all the columns 

df.aggregate(['sum', 'min'])
### We are going to find aggregation for these columns 
This function lets you perform several calculations at once, like getting the total, average, and maximum of a column in one go.

df.aggregate({"Order ID":['sum', 'min'], 
              "Sales":['max', 'min'], 
             }) 
### Common Aggregations with groupby():
Think of it like sorting your data into groups (e.g., by category) and then calculating things like totals or averages for each group.
df.groupby('Category').agg({'Sales': ['sum', 'mean', 'max', 'min']})

### Used to summarize and aggregate data with multiple dimensions.
It’s like an Excel pivot table. You can reorganize your data to show summaries by different categories and calculations.
df.pivot_table(values='Sales', index='Category', aggfunc='sum')
### Apply() – Custom Aggregation Function
Allows applying custom functions to each group.
Lets you write your own function to apply on each row or group of data.
df.groupby('Category')['Sales'].apply(lambda x: x.max() - x.min())

### sum() – Add up values
Adds up all the numbers in a column.
df['Sales'].sum()

### mean() – Calculating Average
Finds the average of a numerical column.
df['Sales'].mean()
### count() – Counting Rows
Counts the number of non-null values.
df['Sub-Category'].count()
 ### median() – Median Value
Calculates the median (middle value) of a column.
df['Sales'].median()
### std() – Standard Deviation
Computes the standard deviation of a numerical column.
df['Sales'].std()
### var() – Variance Calculation
Computes the variance of a numerical column.
df['Sales'].var()
### nunique() – Counting Unique Values
Finds the number of unique values in a column.
df['Category'].nunique()
## Data preprocessing

### isnull() / notnull() – Check for missing values
#### for missing values
df.isnull().sum() 
#### for non missing values
df.isnull().sum() 
#### Remove rows with any missing value
df.dropna()
#### Remove columns with missing values
df.dropna(axis=1)
#### fillna() – Fill missing values
Fills missing values with a specified value or strategy.
df.fillna(0)  # Replace missing values with 0

### Handling Duplicates
#### duplicated() – Identify duplicate rows

df.duplicated().sum()  # Count duplicate rows
#### Checks for duplicate rows in the DataFrame
df.drop_duplicates()  # Remove duplicate rows

### Data Transformation
#### astype() – Convert data types
Converts the data type of columns to a specific type.
df['Customer Name'] = df['Customer Name'].map({'Male': 1, 'Female': 0})  # Convert categorical values to numeric
### rename() – Rename columns
Renames column names for better understanding.
df.rename(columns={'Customer Name': 'name'}, inplace=True)

### Dealing with Outliers
clip() – Limit values within a range
Restrict values to a specified range.
df['Sales'] = df['Sales'].clip(lower=30000, upper=100000)
q1 = df['Sales'].quantile(0.25)
q3 = df['Sales'].quantile(0.75)
iqr = q3 - q1
### Data Encoding
get_dummies() – One-hot encoding
Converts categorical data into dummy variables.
pd.get_dummies(df, columns=['name'])
### adding column
import pandas as pd
import numpy as np
df = pd.DataFrame({'Temperature': ['Hot', 'Cold', 'Warm', 'Cold'],
				})
print(df)
pd.get_dummies(df)

 ### factorize() – Convert categorical values to numeric codes
Encodes labels into numeric values.
import pandas as pd
df = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'B', 'C', 'A']})
df['Category_Code'] = df['Category'].astype('category').cat.codes

print(df)
### Data plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df['Category'].value_counts().plot(kind='pie', autopct='%1.1f%%', title='Sales Category Distribution')
plt.ylabel('')  # Remove the default y-label
plt.show()

df['City'].value_counts().plot(kind="bar")
df['City'].value_counts().plot(kind="area")
df['City'].value_counts().plot(kind="hist")
df['City'].value_counts().plot(kind="kde")
df['City'].value_counts().plot(kind="line")
df['City'].value_counts().plot(kind="area")
## saving data
df = pd.DataFrame({'Name': ['ali', 'hamza', 'farhat'],'Roll no':['2340','23412','1233'], 
                   'marks':['77','34',99], 'age':['20','21','22'], 'city':['lahore','islamabad','karachi'],
                   'country':['pakistan','pakistan','pakistan']})
print(df)


df.to_csv('tutorial.csv')
