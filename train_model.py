import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load your transaction data
df = pd.read_csv("past_customer_transactions.csv")

# 2. Aggregating transaction history per customer to create ML features
features = df.groupby('Customer_ID').agg(
    total_billed=('Billed_Amount', 'sum'),
    total_paid=('Paid_Amount', 'sum'),
    total_outstanding=('Outstanding_Balance', 'sum'),
    avg_billed=('Billed_Amount', 'mean'),
    unpaid_count=('Payment_Behavior', lambda x: (x == 'Unpaid').sum()),
    partially_paid_count=('Payment_Behavior', lambda x: (x == 'Partially Paid').sum()),
    fully_paid_count=('Payment_Behavior', lambda x: (x == 'Fully Paid').sum())
).reset_index()

# 3. Create Feature: Repayment Ratio
features['repayment_ratio'] = np.where(
    features['total_billed'] > 0,
    features['total_paid'] / features['total_billed'],
    1.0
)

# 4. Define Target Variable: Risk Label
# Low Risk (1) vs High Risk / Default (0)
def assign_risk(row):
    if row['repayment_ratio'] >= 0.70 and row['unpaid_count'] <= 2:
        return 1  # Good Credit / Low Risk
    else:
        return 0  # Bad Credit / High Risk

features['credit_status'] = features.apply(assign_risk, axis=1)

# 5. Prepare data for model training
X = features[[
    'total_billed', 'total_paid', 'total_outstanding', 
    'avg_billed', 'unpaid_count', 'partially_paid_count', 
    'fully_paid_count', 'repayment_ratio'
]]
y = features['credit_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 7. Save model to disk
joblib.dump(model, "credit_model.pkl")
print("✅ Machine learning model saved as credit_model.pkl!")