# ===============================
# BASIC VERSION (FOR REFERENCE)
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Basic version setup
np.random.seed(0)
import tensorflow as tf
tf.random.set_seed(1)

housing = pd.read_csv('Hyderabad.csv')
features = housing.drop(['Location', 'Price'], axis=1)
labels = housing['Price']

model = Sequential()
model.add(Dense(38, activation='relu', input_shape=(38,)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=[keras.metrics.RootMeanSquaredError()]
)

model.fit(features, labels, epochs=10, batch_size=10)
model.evaluate(features, labels)
predictions = model.predict(features)
"""

# ===============================
# IMPROVED VERSION (ACTIVE)
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
import tensorflow as tf
tf.random.set_seed(1)

# Load dataset
housing = pd.read_csv('./Hyderabad.csv')
features = housing.drop(['Location', 'Price'], axis=1)
labels = housing['Price']

# Clean unusual placeholder values
features = features.replace(9, 0)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Scale target variable (price)
price_scaler = StandardScaler()
y_train_scaled = price_scaler.fit_transform(y_train.values.reshape(-1,1))
y_test_scaled = price_scaler.transform(y_test.values.reshape(-1,1))

# Build neural network model
model = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

# Compile the model
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=[keras.metrics.RootMeanSquaredError()]
)

# Train the model
history = model.fit(
    x_train, y_train_scaled,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate on test set
test_loss, test_rmse = model.evaluate(x_test, y_test_scaled)
print(f"Test RMSE (scaled): {test_rmse}")

# Make predictions
predictions_scaled = model.predict(x_test)
predictions_original = price_scaler.inverse_transform(predictions_scaled)

# Print first 5 predictions vs actual prices
print("Predictions:", predictions_original[:5].flatten())
print("Actuals:   ", y_test.values[:5])