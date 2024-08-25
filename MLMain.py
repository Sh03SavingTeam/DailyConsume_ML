import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('C:/Users/HyunSang/Desktop/tset.csv')

def preprocess_data(df):
    columns = ['store_reg_num', 'store_name', 'pay_amount']
    df = df[columns]

    # 라벨링
    le_store_reg_num = LabelEncoder()
    le_store_name = LabelEncoder()

    df['store_reg_num'] = le_store_reg_num.fit_transform(df['store_reg_num'])
    df['store_name'] = le_store_name.fit_transform(df['store_name'])

    # 특성과 레이블을 분리
    X = df[['store_reg_num', 'store_name']]
    y = df['pay_amount']

    return X, y, le_store_reg_num, le_store_name

X, y, le_store_reg_num, le_store_name = preprocess_data(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_model():
    input_store_reg_num = Input(shape=(1,), name='store_reg_num')
    input_store_name = Input(shape=(1,), name='store_name')

    embed_store_reg_num = Embedding(input_dim=len(le_store_reg_num.classes_), output_dim=5)(input_store_reg_num)
    embed_store_name = Embedding(input_dim=len(le_store_name.classes_), output_dim=5)(input_store_name)

    flatten_store_reg_num = Flatten()(embed_store_reg_num)
    flatten_store_name = Flatten()(embed_store_name)

    concat = Concatenate()([flatten_store_reg_num, flatten_store_name])

    dense1 = Dense(64, activation='relu')(concat)
    dense2 = Dense(32, activation='relu')(dense1)
    output = Dense(1)(dense2)

    model = Model(inputs=[input_store_reg_num, input_store_name], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

model = create_model()

history = model.fit(
    [X_train['store_reg_num'], X_train['store_name']],
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

loss = model.evaluate([X_test['store_reg_num'], X_test['store_name']], y_test)
print(f'Test loss: {loss}')

predictions = model.predict([X_test['store_reg_num'], X_test['store_name']])
print(predictions)