import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

# Load the training dataset
train_data = pd.read_csv(
    'C:/Users/Nec/Desktop/EXTRA CURR/zips/NAD/scripts/Train_data.csv')

# Load the test dataset
test_data = pd.read_csv(
    'C:/Users/Nec/Desktop/EXTRA CURR/zips/NAD/scripts/test_data.csv')

# Extract features and target variable from the datasets
X_train = train_data.drop(columns=['xAttack'])
y_train = train_data['xAttack']

X_test = test_data.drop(columns=['xAttack'])
y_test = test_data['xAttack']

# Remove 'Unnamed: 0' column if present
if 'Unnamed: 0' in X_train.columns:
    X_train = X_train.drop(columns=['Unnamed: 0'])

if 'Unnamed: 0' in X_test.columns:
    X_test = X_test.drop(columns=['Unnamed: 0'])

# Encode categorical variables (if any)
label_encoder = LabelEncoder()
for column in ['protocol_type', 'service', 'flag']:
    X_train[column] = label_encoder.fit_transform(X_train[column])
    X_test[column] = label_encoder.transform(X_test[column])

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encode target variable (assuming xAttack is the target variable)
y_encoder = LabelEncoder()
y_train = y_encoder.fit_transform(y_train)
y_test = y_encoder.transform(y_test)

# Define the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(y_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the trained model
model.save('classifier_model.h5')
print("Model saved successfully.")

# After fitting the LabelEncoder to the target variable during training
np.save('y_encoder_classes.npy', y_encoder.classes_)


# Evaluate the model on the test dataset
y_pred_probabilities = model.predict(X_test)
y_pred = np.argmax(y_pred_probabilities, axis=1)

# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred))
