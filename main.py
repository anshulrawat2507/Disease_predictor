import pandas as pd

df=pd.read_csv("diabetes.csv")

# replacing invalid zeroes with col means(for data cleaning)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col]=df[col].replace(0,df[col].mean())

print("cleaned data set preview ");
print(df.head());

X = df.drop("Outcome", axis=1)  # All columns except target
y = df["Outcome"]               # Target (label)
#nomalise the features(improves ML)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#split into training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Training set size:", len(X_train))
print("Test set size:", len(X_test))




print("\n====== HEART ATTACK DATASET PREVIEW ======")

heart_df = pd.read_csv("heart_attack_prediction_india.csv")

#first few rows
print(heart_df.head())

#info
print("\nHeart Dataset Info:")
print(heart_df.info())

# Summary stats
print("\nHeart Dataset Description:")
print(heart_df.describe())

# Check for missing values
print("\nMissing values in heart dataset:")
print(heart_df.isnull().sum())


# ==== HEART DATASET PREPROCESSING ====

print("\n====== PREPROCESSING HEART ATTACK DATASET ======")

# Drop unnecessary columns (optional but recommended)
heart_df = heart_df.drop(columns=['Patient_ID', 'State_Name', 'Gender'])

# Drop missing values
heart_df.dropna(inplace=True)

# Set correct target column
target_column = 'Heart_Attack_Risk'

# Separate features and target
X_heart = heart_df.drop(target_column, axis=1)
y_heart = heart_df[target_column]

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler_heart = StandardScaler()
X_heart_scaled = scaler_heart.fit_transform(X_heart)

# Train-test split
from sklearn.model_selection import train_test_split
Xh_train, Xh_test, yh_train, yh_test = train_test_split(
    X_heart_scaled, y_heart, test_size=0.2, random_state=42
)

print("Heart dataset training set size:", len(Xh_train))
print("Heart dataset test set size:", len(Xh_test))


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

print("\n====== TRAINING DIABETES MODEL ======")

# Train model for diabetes dataset
diabetes_model = LogisticRegression(max_iter=1000)
diabetes_model.fit(X_train, y_train)
y_pred_diabetes = diabetes_model.predict(X_test)

# Evaluate
accuracy_diabetes = accuracy_score(y_test, y_pred_diabetes)
print("Diabetes Prediction Accuracy:", accuracy_diabetes)

# Save model
joblib.dump(diabetes_model, "diabetes_model.pkl")
print("Diabetes model saved as 'diabetes_model.pkl'")


print("\n====== TRAINING HEART ATTACK MODEL ======")

# Train model for heart attack dataset
heart_model = LogisticRegression(max_iter=1000)
heart_model.fit(Xh_train, yh_train)
y_pred_heart = heart_model.predict(Xh_test)

# Evaluate
accuracy_heart = accuracy_score(yh_test, y_pred_heart)
print("Heart Attack Prediction Accuracy:", accuracy_heart)

# Save model
joblib.dump(heart_model, "heart_model.pkl")
print("Heart model saved as 'heart_model.pkl'")
