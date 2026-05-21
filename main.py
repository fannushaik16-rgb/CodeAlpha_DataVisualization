import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (built-in from seaborn - no download needed!)
df = sns.load_dataset('titanic')

# Check the data
print(df.head())
print(df.shape)

# ── Chart 1: How many survived? ──────────────────────────
plt.figure(figsize=(6,4))
sns.countplot(x='survived', data=df, palette='Set2')
plt.title('Survival Count (0 = No, 1 = Yes)')
plt.savefig('chart1_survival.png')
plt.show()

# ── Chart 2: Survival by Gender ──────────────────────────
plt.figure(figsize=(6,4))
sns.countplot(x='sex', hue='survived', data=df, palette='Set1')
plt.title('Survival by Gender')
plt.savefig('chart2_gender.png')
plt.show()

# ── Chart 3: Age Distribution ────────────────────────────
plt.figure(figsize=(7,4))
sns.histplot(df['age'].dropna(), bins=30, kde=True, color='steelblue')
plt.title('Age Distribution of Passengers')
plt.savefig('chart3_age.png')
plt.show()

# ── Chart 4: Fare by Passenger Class ────────────────────
plt.figure(figsize=(7,4))
sns.boxplot(x='pclass', y='fare', data=df, palette='pastel')
plt.title('Ticket Fare by Passenger Class')
plt.savefig('chart4_fare.png')
plt.show()

# ── Chart 5: Correlation Heatmap ─────────────────────────
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('chart5_heatmap.png')
plt.show()

print("All charts saved!")
input("Press Enter to exit...")