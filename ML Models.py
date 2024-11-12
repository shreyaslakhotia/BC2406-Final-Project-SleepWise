# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
file_path = '/mnt/data/Sleep_Efficiency (1).csv'
sleep_data = pd.read_csv(file_path)

# Step 1: Create the new attribute "Sleep Efficiency Class" based on sleep efficiency percentiles
# Calculate percentiles
percentiles = sleep_data['Sleep efficiency'].quantile([0.35, 0.60])
percentile_35 = percentiles.loc[0.35]
percentile_60 = percentiles.loc[0.60]

# Define the function to categorize sleep efficiency
def categorize_sleep_efficiency(efficiency):
    if efficiency > percentile_60:
        return 3  # Good
    elif efficiency > percentile_35:
        return 2  # Average
    else:
        return 1  # Poor

# Create the new attribute
sleep_data['Sleep Efficiency Class'] = sleep_data['Sleep efficiency'].apply(categorize_sleep_efficiency)

# Step 2: Prepare data for Decision Tree Classifier
# Selecting the features (X) and target (Y)
X = sleep_data[['Age', 'Gender', 'Sleep duration', 'REM sleep percentage',
                'Deep sleep percentage', 'Light sleep percentage', 'Awakenings',
                'Caffeine consumption', 'Alcohol consumption', 'Smoking status', 
                'Exercise frequency']]

y = sleep_data['Sleep Efficiency Class']

# Encoding categorical variables
label_encoder = LabelEncoder()
X['Gender'] = label_encoder.fit_transform(X['Gender'])
X['Smoking status'] = label_encoder.fit_transform(X['Smoking status'])

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train an initial Decision Tree Classifier
initial_tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
initial_tree_clf.fit(X_train, y_train)

# Plotting the decision tree
plt.figure(figsize=(20, 10))
plot_tree(initial_tree_clf, feature_names=X.columns, class_names=['Poor', 'Average', 'Good'], filled=True, rounded=True)
plt.title("Initial Decision Tree for Sleep Efficiency Classification")
plt.show()

# Step 5: Cross-validation to determine optimal number of leaf nodes
leaf_nodes = range(2, 51)
cv_scores = []
for leaf_node in leaf_nodes:
    tree_clf = DecisionTreeClassifier(max_leaf_nodes=leaf_node, random_state=42)
    scores = cross_val_score(tree_clf, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(np.mean(scores))

# Plot the cross-validation scores for different leaf nodes
plt.figure(figsize=(10, 6))
plt.plot(leaf_nodes, cv_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Leaf Nodes')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Optimal Number of Leaf Nodes for Decision Tree')
plt.grid(True)
plt.show()

# Find the optimal number of leaf nodes
optimal_leaf_nodes = leaf_nodes[np.argmax(cv_scores)]

# Step 6: Train the Decision Tree Classifier using optimal leaf nodes
optimal_tree_clf = DecisionTreeClassifier(max_leaf_nodes=optimal_leaf_nodes, random_state=42)
optimal_tree_clf.fit(X_train, y_train)

# Make predictions and evaluate the model
y_optimal_pred = optimal_tree_clf.predict(X_test)
optimal_accuracy = accuracy_score(y_test, y_optimal_pred)
optimal_conf_matrix = confusion_matrix(y_test, y_optimal_pred)
optimal_class_report = classification_report(y_test, y_optimal_pred)

# Step 7: Hyperparameter tuning using GridSearchCV
param_grid = {
    'max_depth': [2, 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Extract the best parameters and retrain the model
best_params = grid_search.best_params_
tuned_tree_clf = DecisionTreeClassifier(**best_params, random_state=42)
tuned_tree_clf.fit(X_train, y_train)

# Make predictions with the tuned model
y_tuned_pred = tuned_tree_clf.predict(X_test)
tuned_accuracy = accuracy_score(y_test, y_tuned_pred)
tuned_conf_matrix = confusion_matrix(y_test, y_tuned_pred)
tuned_class_report = classification_report(y_test, y_tuned_pred)

# Step 8: Plot the tuned decision tree
plt.figure(figsize=(20, 10))
plot_tree(tuned_tree_clf, feature_names=X.columns, class_names=['Poor', 'Average', 'Good'], filled=True, rounded=True)
plt.title("Tuned Decision Tree for Sleep Efficiency Classification (Max Depth = 4)")
plt.show()

# Step 9: Evaluate feature importance for the tuned model
feature_importances = tuned_tree_clf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

# Plot feature importances with numerical values
plt.figure(figsize=(12, 8))
bars = plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
for bar in bars:
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():.3f}', va='center', fontsize=10)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Tuned Decision Tree')
plt.gca().invert_yaxis()
plt.show()
