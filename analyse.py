import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

##### Read dataset
df = pd.read_csv('data.csv')


# #### Analyse data
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
        
# # Print feature correlations    
# print(df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
# print(df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
# print(df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean())


# #### Preprocessing
# # Drop column / row
# df = df.drop(['PassengerId', 'Ticket'], axis=1) # Column
# df = df.drop(0, axis=0)                         # Row #0

# # Drop rows with any NaN value
# df = df.dropna()

# # Fill NaN values
# df['Age'] = df['Age'].fillna(df['Age'].dropna().median())   # Fill 'Age' with median
# df['Age'] = df['Age'].fillna(method='ffill')                # Forward fill ; bfill = backward fill

# # Categorical feature normalization
# df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

# # Scale values
# df[['Age', 'Fare']] = MinMaxScaler().fit_transform(['Age', 'Fare'])

# # Cast values to specified format
# df['Age'] = df['Age'].astype('int')

# # New features (using list comprehension)
# df['Age_child'] = [int(x<18) for x in df['Age’]]
# df['Age_adult'] = [int(x>=18) for x in df['Age’]]
