# ai_model.py
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Port + Service + Risk level (label)
# Very simplified sample data
data = [
    [22, 1],    # SSH - high
    [21, 1],    # FTP - high
    [80, 1],    # HTTP - high
    [443, 0],   # HTTPS - low
    [53, 0],    # DNS - low
    [3306, 1],  # MySQL - high
    [25, 1],    # SMTP - high
    [110, 0],   # POP3 - low
]

X = [[row[0]] for row in data]  # Port
y = [row[1] for row in data]    # Risk

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.joblib")
print("[+] AI model trained and saved!")
