import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
file_path = r"C:\Users\hp-pc\Downloads\test (1).csv"  # Update with your actual file path
df = pd.read_csv(file_path)

df["y"] = ["yes" if x % 2 == 0 else "no" for x in range(len(df))] 


# Convert response column to binary (1 = yes, 0 = no)
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# ----------------------------------------------
# 📊 1. Handling Missing Values
# ----------------------------------------------
print("Missing Values Before Handling:\n", df.isnull().sum())

# Fill missing categorical values with mode
categorical_cols = ["job", "marital", "education", "contact", "poutcome"]
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Fill missing numerical values with median
numerical_cols = ["age", "balance", "duration", "campaign", "previous"]
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

print("\nMissing Values After Handling:\n", df.isnull().sum())

# ----------------------------------------------
# 📊 2. Feature Engineering: Creating New Features
# ----------------------------------------------
# Success rate from past campaigns
df['past_success_rate'] = df['previous'] / (df['campaign'] + 1)  # Avoid division by zero
df['past_success_rate'].fillna(0, inplace=True)  # Replace NaN with 0

# Contact frequency category
df['contact_intensity'] = pd.cut(df['campaign'], bins=[0, 1, 3, 10, np.inf], 
                                 labels=['Very Low', 'Low', 'Medium', 'High'])

# Financial Stability Feature
df['financial_stability'] = np.where(df['balance'] > 5000, 1, 0)  # 1 = Stable, 0 = Not Stable

# ----------------------------------------------
# 📊 3. Encoding Categorical Variables
# ----------------------------------------------
# One-Hot Encoding for categorical variables
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Label Encoding for 'contact_intensity'
df['contact_intensity'] = LabelEncoder().fit_transform(df['contact_intensity'])

# ----------------------------------------------
# 📊 4. Feature Scaling (Standardization)
# ----------------------------------------------
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# ----------------------------------------------
# 📊 5. Splitting Data into Train & Test Sets
# ----------------------------------------------
X = df.drop(columns=['y'])  # Features
y = df['y']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nDataset Shape After Preprocessing:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)

# Save the preprocessed data
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\n Data Preprocessing Completed & Saved!")
