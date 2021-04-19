import pandas as pd
from matplotlib import pyplot as plt


##### Read dataset
df = pd.read_csv('data.csv')


##### Analyse data
# # Print head / tail of dataset
# print("Head:\n", df.head(n=5))
# print("\n")
# print("Tail:\n", df.tail(n=10))

# # Get column names
# print("Column names: ", df.columns)

# # Describe dataframe
# print(df.describe())

# # Check if dataframe has NaN values
# print(df.isna())
# print(df.isna().any())
        
# Print feature correlations    
# print(df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
# print(df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
print(df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean())
