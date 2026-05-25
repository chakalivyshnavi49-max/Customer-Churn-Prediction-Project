import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {
    'Age':[25,45,30,35,50,23,40,60,48,33],
    'MonthlyCharges':[50,80,60,70,90,55,75,95,85,65],
    'Tenure':[12,24,8,15,30,10,20,40,28,14],
    'Contract':['Month-to-month','One year','Month-to-month','Two year','One year',
                'Month-to-month','Two year','One year','Two year','Month-to-month'],
    'Churn':['Yes','No','Yes','No','No','Yes','No','No','No','Yes']
}

df = pd.DataFrame(data)

le_contract = LabelEncoder()
le_churn = LabelEncoder()
df['Contract'] = le_contract.fit_transform(df['Contract'])
df['Churn'] = le_churn.fit_transform(df['Churn'])

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,4))
plt.imshow(cm)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i,j], ha='center', va='center')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

results = X_test.copy()
results['Actual Churn'] = y_test.values
results['Predicted Churn'] = y_pred
results.to_excel('churn_predictions.xlsx', index=False)
print('Predictions saved to churn_predictions.xlsx')
