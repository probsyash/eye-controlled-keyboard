import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

classifier = RandomForestClassifier(n_estimators=100, random_state=42)

file = open("dataStorage/landmarks.txt", "r")
x_data = [] # feature data
y_data = [] # label data

index = 0
while True:
    line = file.readline()
    if not line:
        break

    coords = line.strip().split(",")
    row = [float(coord) for coord in coords[:-1]]  # Convert all but the last element to float

    x_data.append(row)
    y_data.append(coords[-1]) # Last element is the label

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
accuracy_score(y_test, y_pred)
classification_report(y_test, y_pred)
confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))    
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))