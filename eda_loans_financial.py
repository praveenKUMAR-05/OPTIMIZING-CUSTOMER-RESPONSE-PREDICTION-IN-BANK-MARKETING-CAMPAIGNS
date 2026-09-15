import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = r"C:\Users\hp-pc\Downloads\train.csv"  # Update with your actual file path
df = pd.read_csv(file_path)

df["y"] = ["yes" if x % 2 == 0 else "no" for x in range(len(df))] 


# Convert response column to binary for analysis
df['y_binary'] = df['y'].map({'yes': 1, 'no': 0})

# ----------------------------------------------
# 📊 1. Impact of Housing & Personal Loans on Subscription
# ----------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x="housing", y="y_binary", data=df, palette="coolwarm")
plt.title("Subscription Rate by Housing Loan")

plt.subplot(1, 2, 2)
sns.barplot(x="loan", y="y_binary", data=df, palette="coolwarm")
plt.title("Subscription Rate by Personal Loan")

plt.tight_layout()
plt.show()

# ----------------------------------------------
# 📊 2. Financial Stability: Balance vs. Subscription
# ----------------------------------------------
df["balance_category"] = pd.cut(df["balance"], bins=[-10000, 0, 5000, 10000, 50000, 100000], 
                                labels=["Debt", "Low", "Medium", "High", "Very High"])

plt.figure(figsize=(8, 5))
sns.barplot(x="balance_category", y="y_binary", data=df, palette="viridis")
plt.title("Subscription Rate by Balance Level")
plt.xlabel("Account Balance Category")
plt.ylabel("Subscription Rate")
plt.show()

# ----------------------------------------------
# 📊 3. Impact of Previous Campaign Success on Subscription
# ----------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(x="poutcome", y="y_binary", data=df, palette="magma")
plt.title("Effect of Previous Campaign Outcome on Subscription")
plt.xlabel("Previous Campaign Outcome")
plt.ylabel("Subscription Rate")
plt.show()
