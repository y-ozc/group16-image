import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.utils import to_categorical

df_0 = pd.read_csv('data/0_processed.csv')
df_1 = pd.read_csv('data/1_processed.csv')
df_2 = pd.read_csv('data/2_processed.csv')
df_3 = pd.read_csv('data/3_processed.csv')
df_4 = pd.read_csv('data/4_processed.csv')
df_5 = pd.read_csv('data/5_processed.csv')
df_6 = pd.read_csv('data/6_processed.csv')
df_7 = pd.read_csv('data/7_processed.csv')
df_8 = pd.read_csv('data/8_processed.csv')
df_9 = pd.read_csv('data/9_processed.csv')

df = pd.concat([
    df_0, df_1, df_2, df_3, df_4,
    df_5, df_6, df_7, df_8, df_9
], axis=0, ignore_index=True)
train_df = (df.drop(["id"], axis=1)).copy()

x = train_df.iloc[:, 1:]  
y = train_df.iloc[:, 0]

#print(train_df.shape)
#print(x.shape)

if not isinstance(x, pd.DataFrame):
    x = pd.DataFrame(x)
x = x.apply(pd.to_numeric, errors='coerce')
x = x.fillna(0)  
x = x.values / 255.0
x = x.reshape(-1, 32, 32, 1)
y = to_categorical(y, num_classes=10)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=104, test_size=0.20, shuffle=True)

model = Sequential([
    Input(shape=(32, 32, 1)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

val_loss, val_accuracy = model.evaluate(x_test, y_test)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# ===== SAVE MODEL =====
import os
from keras.models import load_model

# create folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# save full model (recommended format)
model_path = "models/digit_model.keras"
model.save(model_path)

print(f"Model saved to: {model_path}")

# ===== OPTIONAL: LOAD TEST (to verify it works) =====
loaded_model = load_model(model_path)
loss, acc = loaded_model.evaluate(x_test, y_test)
print(f"Loaded model accuracy: {acc * 100:.2f}%")