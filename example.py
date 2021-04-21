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
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Read data
df = pd.read_csv('data.csv')

# Drop some columns
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Fill 'Age' with median
df['Age'] = df['Age'].fillna(df['Age'].dropna().median())
df['Age'] = df['Age'].astype('int')

# Fill 'Embarked' with top value
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].describe().top)

# Encode values
encoder = LabelEncoder()
df['Sex'] = encoder.fit_transform(df['Sex'])

# Add custom columns
df = df.join(pd.get_dummies(df.Embarked, prefix="Embarked").astype(int))
df = df.drop(['Embarked'], axis=1)

# Separate X, Y
X = df.drop('Survived', axis=1).values
Y = pd.get_dummies(df['Survived'])

# Scale values
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split for train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

# Model
model = Sequential()
model.add(Dense(32, input_shape=(9, ), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.004), metrics=['accuracy'])
history = model.fit(x=X_train, y=Y_train, epochs=200, batch_size=16,
                    validation_split=0.2, callbacks=[EarlyStopping(monitor='loss', patience=2)])


preds = model.predict(X_test)
score = accuracy_score([np.argmax(x) for x in preds], [np.argmax(x) for x in Y_test.values])

