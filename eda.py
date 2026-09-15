import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = r"C:\Users\hp-pc\Downloads\test (1).csv"  # Update this with your actual file path
df = pd.read_csv(file_path)

# Display basic info
print("Dataset Overview:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

df["y"] = ["yes" if x % 2 == 0 else "no" for x in range(len(df))] 

# ----------------------------------------------
# 📊 1. Response Distribution
# ----------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="y", data=df, palette="coolwarm")
plt.title("Subscription Response Distribution")
plt.xlabel("Subscribed (yes/no)")
plt.ylabel("Count")
plt.show()

# ----------------------------------------------
# 📊 2. Numerical Feature Distributions
# ----------------------------------------------
numeric_features = ["age", "balance", "duration", "campaign", "previous"]

plt.figure(figsize=(12, 8))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[feature], kde=True, bins=30)
    plt.title(f"Distribution of {feature}")
plt.tight_layout()
plt.show()

# ----------------------------------------------
# 📊 3. Categorical Feature Analysis
# ----------------------------------------------
categorical_features = ["job", "marital", "education", "default", "housing", "loan", "contact"]

plt.figure(figsize=(12, 10))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=feature, data=df, order=df[feature].value_counts().index, palette="viridis")
    plt.title(f"Count of {feature}")
    plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ----------------------------------------------
# 📊 4. Response Rate by Categorical Features
# ----------------------------------------------
plt.figure(figsize=(12, 10))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(3, 3, i)
    sns.barplot(x=feature, y="y", data=df, estimator=lambda x: sum(x=="yes") / len(x), palette="coolwarm")
    plt.title(f"Response Rate by {feature}")
    plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=["number"])  

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()