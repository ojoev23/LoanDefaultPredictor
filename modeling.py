import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load model data
print("Loading model data")
df = pd.read_csv('lending_club_model_ready.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split into test and train for X and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Final splits
y_pred = rf.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            linewidths=1, linecolor='black',
            xticklabels=['Predicted Good Loan', 'Predicted Default'],
            yticklabels=['Actual Good Loan', 'Actual Default'])

plt.title('Credit Risk Model: Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('What the Model Predicted', fontsize=12, fontweight='bold')
plt.ylabel('The Actual Truth', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('portfolio_confusion_matrix.png', dpi=300)
print("Saved 'portfolio_confusion_matrix.png'")
plt.show()


# Classification Report
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
print(report_df)
metrics_to_plot = report_df.drop('support', axis=1)

plt.figure(figsize=(8, 5))
sns.heatmap(metrics_to_plot, annot=True, cmap='RdYlGn', vmin=0, vmax=1,
            linewidths=1, linecolor='black')

plt.title('Model Performance Report Card', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Metric Category', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('portfolio_classification_report.png', dpi=300)
print("Saved 'portfolio_classification_report.png'")
plt.show()