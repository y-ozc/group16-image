import numpy as np
import pandas as pd
from keras.models import load_model

# load the data into the format into the correct form for kaggle
df = pd.read_csv("unlabeled.csv")

ids = df["id"]

x = df.drop(["id"], axis=1)

x = x.apply(pd.to_numeric, errors='coerce')
x = x.fillna(0)

x = x.values / 255.0
x = x.reshape(-1, 32, 32, 1)

model = load_model("models/digit_model_cnn.keras")

predictions = model.predict(x)
labels = np.argmax(predictions, axis=1)

output = pd.DataFrame({
    "id": ids,
    "label": labels
})

output.to_csv("predictions.csv", index=False)

print("saved")