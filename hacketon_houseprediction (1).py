# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # --- STEP 1: Generate Dummy Data (Replace this block with pd.read_csv if you have a file) ---
# # Creating a synthetic dataset for demonstration purposes
# np.random.seed(42)
# data_size = 1000
# sqft_living = np.random.randint(500, 5000, size=data_size)
# # Price = roughly $300 per sqft + random noise
# price = sqft_living * 300 + np.random.normal(0, 50000, size=data_size)

# df = pd.DataFrame({
#     'sqft_living': sqft_living,
#     'price': price
# })

# # If you have a real file, uncomment the line below and comment out the synthetic data generation above
# # df = pd.read_csv("data.csv")

# # --- STEP 2: Preprocessing ---
# # Inspecting the data
# print("Data Shape:", df.shape)
# print(df.head())

# # Select features (X) and target (y)
# # We use the full dataset here, not just the first 500 rows
# X = df[['sqft_living']] 
# y = df['price']

# # --- STEP 3: Visualization ---
# plt.figure(figsize=(10, 6))
# plt.scatter(X, y, alpha=0.5, color='blue')
# plt.title("House Price vs. Square Footage")
# plt.xlabel("Sqft Living")
# plt.ylabel("Price")
# plt.grid(True)
# plt.show()

# # --- STEP 4: Split Data ---
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42
# )

# # --- STEP 5: Train Model ---
# model = LinearRegression()
# model.fit(X_train, y_train)

# # --- STEP 6: Evaluation ---
# y_pred = model.predict(X_test)

# print("\n--- Model Evaluation ---")
# print(f"MAE : {mean_absolute_error(y_test, y_pred):,.2f}")
# print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
# print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

# # --- STEP 7: Prediction ---
# # Predict price for a realistic house size (e.g., 2000 sqft) rather than 1 sqft
# new_house_size = 2000
# new_house = pd.DataFrame([[new_house_size]], columns=["sqft_living"])
# predicted_price = model.predict(new_house)

# print(f"\nPredicted Price for a {new_house_size} sqft house: ${predicted_price[0]:,.2f}")






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# --- STEP 1: Generate Dummy Data with ONLY essential features ---
np.random.seed(42)
data_size = 1000

# Generate realistic data
sqft_living = np.random.randint(500, 5000, size=data_size)
bedrooms = np.random.randint(1, 6, size=data_size)
bathrooms = np.random.uniform(1, 4, size=data_size).round(1)
condition = np.random.randint(1, 6, size=data_size)

# Price calculation: base price + features
base_price = sqft_living * 300  # $300 per sqft
bedroom_bonus = bedrooms * 15000  # $15k per bedroom
bathroom_bonus = bathrooms * 20000  # $20k per bathroom
condition_bonus = (condition - 3) * 25000  # Condition adjustment

price = base_price + bedroom_bonus + bathroom_bonus + condition_bonus + np.random.normal(0, 50000, size=data_size)

df = pd.DataFrame({
    'sqft_living': sqft_living,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'condition': condition,
    'price': price
})

print("📊 Dataset created with shape:", df.shape)
print(df.head())

# --- STEP 2: Preprocessing ---
X = df[['sqft_living', 'bedrooms', 'bathrooms', 'condition']]
y = df['price']

print("\n📈 Features used for training:")
print(X.columns.tolist())

# --- STEP 3: Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# --- STEP 4: Train Model ---
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\n✅ Model trained successfully!")
print(f"Model coefficients: {model.coef_}")
print(f"Model intercept: {model.intercept_:,.2f}")

# --- STEP 5: Evaluation ---
y_pred = model.predict(X_test)

print("\n📊 --- Model Evaluation ---")
print(f"MAE : ${mean_absolute_error(y_test, y_pred):,.2f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

# --- STEP 6: Save Model ---
MODEL_FILENAME = 'house_price_model_simple.pkl'
joblib.dump(model, MODEL_FILENAME)
print(f"\n💾 Model saved as '{MODEL_FILENAME}'")

# --- STEP 7: Test Prediction ---
print("\n🧪 --- Test Predictions ---")
test_cases = [
    [2000, 3, 2.0, 3],  # 2000 sqft, 3 bed, 2 bath, average condition
    [1500, 2, 1.5, 4],  # 1500 sqft, 2 bed, 1.5 bath, good condition
    [3000, 4, 3.0, 2],  # 3000 sqft, 4 bed, 3 bath, fair condition
]

for i, test in enumerate(test_cases):
    test_df = pd.DataFrame([test], columns=['sqft_living', 'bedrooms', 'bathrooms', 'condition'])
    prediction = model.predict(test_df)[0]
    print(f"Case {i+1}: {test[0]} sqft, {test[1]} bed, {test[2]} bath, condition {test[3]}")
    print(f"   → Predicted price: ${prediction:,.2f}")

# --- STEP 8: Visualization ---
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.scatter(X_test['sqft_living'], y_test, alpha=0.5, color='blue', label='Actual')
plt.scatter(X_test['sqft_living'], y_pred, alpha=0.5, color='red', label='Predicted')
plt.title("Price vs Square Footage")
plt.xlabel("Sqft Living")
plt.ylabel("Price")
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(X_test['bedrooms'], y_test, alpha=0.5, color='blue')
plt.title("Price vs Bedrooms")
plt.xlabel("Bedrooms")
plt.ylabel("Price")

plt.subplot(2, 2, 3)
plt.scatter(X_test['bathrooms'], y_test, alpha=0.5, color='green')
plt.title("Price vs Bathrooms")
plt.xlabel("Bathrooms")
plt.ylabel("Price")

plt.subplot(2, 2, 4)
plt.scatter(X_test['condition'], y_test, alpha=0.5, color='purple')
plt.title("Price vs Condition")
plt.xlabel("Condition (1-5)")
plt.ylabel("Price")

plt.tight_layout()
plt.show()

print("\n🎯 Training complete! Use the saved model in backend.py")