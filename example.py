# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:16:31 2021

@author: Pawel
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.neighbors import KNeighborsClassifier

def check(a, b='Survived'):
    print(df[[a, b]].groupby([a], as_index=False).mean())

# Read data
df = pd.read_csv('data.csv')

# Drop some columns
df = df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)

# Fill 'Age' with median
df['Age'] = df['Age'].fillna(df['Age'].dropna().median())
df['Age'] = df['Age'].astype('int')

# Fill 'Embarked' with top value
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].describe().top)

# Encode values
encoder = LabelEncoder()
df['Sex'] = encoder.fit_transform(df['Sex'])

### Add custom columns
# Embarked
df = df.join(pd.get_dummies(df.Embarked, prefix="Embarked").astype(int))
df = df.drop(['Embarked'], axis=1)
# Pclass
df = df.join(pd.get_dummies(df['Pclass'], prefix='Pclass').astype('int'))
df = df.drop(['Pclass'], axis=1)
# Fare
df['low_fare'] = [int(x < df['Fare'].describe()['25%']) for x in df['Fare']]
df['high_fare'] = [int(x > df['Fare'].describe()['75%']) for x in df['Fare']]
# df['normal_fare'] = [not df['low_fare'][i] and not df['high_fare'][i] for i in range(len(df))]
# df['normal_fare'] = df['normal_fare'].astype('int')
df = df.drop(['Fare'], axis=1)
# Title
df['title_mrs'] = [int('Mrs.' in x) for x in df['Name']]
df['title_mr'] = [int('Mr.' in x) for x in df['Name']]
df['title_miss'] = [int('Miss.' in x) for x in df['Name']]
df = df.drop(['Name'], axis=1)
# Age
df['age_young'] = [int(x < df['Age'].describe()["25%"]) for x in df['Age']]
df['age_old'] = [int(x > df['Age'].describe()["75%"]) for x in df['Age']]
df['age_middle'] = [int(not df['age_young'][i] and not df['age_old'][i]) for i in range(len(df))]
df = df.drop(['Age'], axis=1)

# Separate X, Y
X = df.drop('Survived', axis=1).values
Y = df['Survived']
#Y = pd.get_dummies(df['Survived'])

# Scale values
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split for train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, shuffle=False)

# Model
model = Sequential()
model.add(Dense(16, input_shape=(len(X_train[0]), ), activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(8, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])
history = model.fit(x=X_train, y=Y_train, epochs=200, batch_size=16,
                    validation_split=0.1, callbacks=[EarlyStopping(monitor='loss', patience=4)])


preds = model.predict(X_test)
preds = preds.reshape(179, )
preds = np.round(preds)
score = accuracy_score(preds, Y_test)
print(f'Score for MLP: {score}')

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
preds = knn.predict(X_test)
score_knn = accuracy_score(preds, Y_test)
print(f'Score for KNN: {score_knn}')

