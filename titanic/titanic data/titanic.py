

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
gender = pd.read_csv("gender_submission.csv")
subset = train[["PassengerId", "Sex"]]
print("TRAIN SHAPE:", train.shape)
print("TEST SHAPE:", test.shape)
print("GENDER SHAPE:", gender.shape)
print("-" * 40)



#missing value
print("missing value")
print(train.isnull().sum())
print("original shape ",train.shape)

#handle missing data by dropping any rows with missing age and embarked
cleantrain=train.dropna(subset=['Age','Embarked'])
print("shape after ckeaning data" ,cleantrain.shape)




# ========== 2. First few rows ==========
print("=== HEAD ===")
print(cleantrain.head(), "\n")

# ========== 3. Basic info (missing, types) ==========
print("=== INFO ===")
print(cleantrain.info(), "\n")

# ========== 4. Summary statistics ==========
print("=== NUMERIC SUMMARY ===")
print(cleantrain.describe(), "\n")

print("=== CATEGORICAL SUMMARY ===")
print(cleantrain.describe(include='object'), "\n")

# ========== 5. Missing values ==========
print("=== MISSING VALUES ===")
print(cleantrain.isnull().sum(), "\n")

# ========== 6. Unique values ==========
print("=== UNIQUE VALUES ===")
print(cleantrain.nunique(), "\n")

# ========== 7. Target distribution ==========
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=cleantrain)
plt.title("Survival Distribution")
plt.show()

# ========== 8. Survival by Sex ==========
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=cleantrain)
plt.title("Survival by Sex")
plt.show()

# ========== 9. Survival by Class ==========
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", hue="Survived", data=cleantrain)
plt.title("Survival by Passenger Class")
plt.show()

# ========== 10. Age distribution ==========
plt.figure(figsize=(6,4))
sns.histplot(cleantrain["Age"], kde=True)
plt.title("Age Distribution")
plt.show()

# ========== 11. Fare distribution ==========
plt.figure(figsize=(6,4))
sns.histplot(cleantrain["Fare"], kde=True, bins=30)
plt.title("Fare Distribution")
plt.show()

# ========== 12. Correlation heatmap ==========
plt.figure(figsize=(10,8))
sns.heatmap(cleantrain.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


