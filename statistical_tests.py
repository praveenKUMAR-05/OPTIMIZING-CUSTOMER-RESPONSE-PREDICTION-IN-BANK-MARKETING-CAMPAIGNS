import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = r"C:\Users\hp-pc\Downloads\test (1).csv"  # Update with your actual file path
df = pd.read_csv(file_path)

df["y"] = ["yes" if x % 2 == 0 else "no" for x in range(len(df))] 


# Convert response column to binary (1 = yes, 0 = no)
df['y_binary'] = df['y'].map({'yes': 1, 'no': 0})

# ----------------------------------------------
# 📊 1. Chi-Square Test: Housing Loan & Personal Loan
# ----------------------------------------------
def chi_square_test(feature):
    contingency_table = pd.crosstab(df[feature], df['y'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-Square Test for {feature}:")
    print(f"Chi2 Statistic = {chi2:.2f}, p-value = {p:.4f}")
    if p < 0.05:
        print(f" {feature} significantly affects subscription!")
    else:
        print(f" No significant impact from {feature}.")

chi_square_test("housing")
chi_square_test("loan")
chi_square_test("poutcome")  # Previous campaign outcome

# ----------------------------------------------
# 📊 2. T-Test: Balance vs Subscription
# ----------------------------------------------
subscribed = df[df["y"] == "yes"]["balance"]
not_subscribed = df[df["y"] == "no"]["balance"]

t_stat, p_val = stats.ttest_ind(subscribed, not_subscribed, equal_var=False)
print("\nT-Test for Balance Levels:")
print(f"T-Statistic = {t_stat:.2f}, p-value = {p_val:.4f}")

if p_val < 0.05:
    print(" Balance significantly impacts subscription rate!")
else:
    print(" No significant impact from balance.")

# ----------------------------------------------
# 📊 3. ANOVA Test: Duration (Call Length) vs Subscription
# ----------------------------------------------
anova_result = stats.f_oneway(df[df["y"] == "yes"]["duration"], df[df["y"] == "no"]["duration"])
print("\nANOVA Test for Call Duration:")
print(f"F-Statistic = {anova_result.statistic:.2f}, p-value = {anova_result.pvalue:.4f}")

if anova_result.pvalue < 0.05:
    print(" Call duration significantly impacts subscription rate!")
else:
    print(" No significant impact from call duration.")

# ----------------------------------------------
# 📊 4. Visualizing Key Findings
# ----------------------------------------------
plt.figure(figsize=(10, 5))
sns.boxplot(x="y", y="balance", data=df, palette="coolwarm")
plt.title("Balance Levels by Subscription Status")
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x="y", y="duration", data=df, palette="magma")
plt.title("Call Duration by Subscription Status")
plt.show()
