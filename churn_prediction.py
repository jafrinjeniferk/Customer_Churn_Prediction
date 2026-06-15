import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("customer_churn.csv")

print("Dataset Loaded Successfully!\n")
print(df.head())

# Convert text columns to numbers
le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["Contract"] = le.fit_transform(df["Contract"])

# Features and Target
X = df.drop(["CustomerID", "Churn"], axis=1)
y = df["Churn"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# User Input
while True:

    print("\nEnter Customer Details")

    gender = input("Gender (Male/Female): ")
    age = int(input("Age: "))
    tenure = int(input("Tenure (months): "))
    monthly_charges = float(input("Monthly Charges: "))
    contract = input("Contract (Monthly/Yearly): ")
    satisfaction = int(input("Satisfaction Score (1-10): "))

    gender = le.fit_transform([gender])[0]
    contract = le.fit_transform([contract])[0]

    customer = [[
        gender,
        age,
        tenure,
        monthly_charges,
        contract,
        satisfaction
    ]]

    prediction = model.predict(customer)

    if prediction[0] == 1:
        print("\nPrediction: Customer Likely To Churn")
    else:
        print("\nPrediction: Customer Likely To Stay")

    choice = input("\nCheck another customer? (yes/no): ")

    if choice.lower() != "yes":
        break