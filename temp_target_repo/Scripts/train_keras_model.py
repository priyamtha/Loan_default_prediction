import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight
import joblib

# ------------------------
# 🔹 Load Data
# ------------------------
df = pd.read_excel("loan_data.xlsb", engine='pyxlsb')  # Replace with your cleaned file if needed
X = df.drop(columns=["TARGET"])
y = df["TARGET"]

# ------------------------
# 🔹 Preprocessing
# ------------------------
X = pd.get_dummies(X)
features = X.columns.tolist()

# Save features for later use in Streamlit
joblib.dump(features, "features.pkl")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# Class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
class_weights_dict = dict(zip(np.unique(y), class_weights))

# ------------------------
# 🔹 SMOTE for balancing
# ------------------------
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# ------------------------
# 🔹 Train-test split
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# ------------------------
# 🔹 Build Keras Model
# ------------------------
model = Sequential()
model.add(Dense(256, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ------------------------
# 🔹 Train Model
# ------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=256,
    class_weight=class_weights_dict,
    callbacks=[early_stop],
    verbose=1
)

# ------------------------
# 🔹 Save Model
# ------------------------
model.save("keras_model.h5")
print("✅ Keras model saved as keras_model.h5")
