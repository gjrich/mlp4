import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Section 1: Import and Inspect the Data
def load_and_inspect_data(filepath):
    """Load mushroom dataset and perform initial inspection"""
    # Reading data with no header
    column_names = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                   'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                   'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                   'stalk-surface-below-ring', 'stalk-color-above-ring',
                   'stalk-color-below-ring', 'veil-type', 'veil-color',
                   'ring-number', 'ring-type', 'spore-print-color',
                   'population', 'habitat']
    
    df = pd.read_csv(filepath, header=None, names=column_names)
    
    print("Dataset Dimensions:", df.shape)
    print("\nFirst 10 rows of the dataset:")
    print(df.head(10))
    
    print("\nSummary Statistics:")
    # All columns are categorical, so we'll check unique values
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")
    
    print("\nCheck for missing values:")
    print(df.isnull().sum())
    
    print("\nValue counts for target variable (class):")
    print(df['class'].value_counts())
    
    return df

# Section 2: Data Exploration and Preparation
def explore_and_prepare_data(df):
    """Explore data distributions and prepare for modeling"""
    # Visualize class distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='class', data=df)
    plt.title('Distribution of Mushroom Classes (Edible vs Poisonous)')
    plt.xlabel('Class (e=edible, p=poisonous)')
    plt.ylabel('Count')
    plt.show()
    
    # Explore a few key features and their relationship to the target
    important_features = ['odor', 'gill-color', 'spore-print-color']
    
    for feature in important_features:
        plt.figure(figsize=(12, 6))
        sns.countplot(x=feature, hue='class', data=df)
        plt.title(f'Distribution of {feature} by Class')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Class', labels=['Edible', 'Poisonous'])
        plt.tight_layout()
        plt.show()

    # Convert categorical variables to numeric
    # First, make a copy to avoid modifying the original
    df_encoded = df.copy()
    
    # Apply Label Encoding to all columns
    label_encoders = {}
    for column in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le
    
    # Return both original and encoded dataframes
    return df, df_encoded, label_encoders

# Section 3: Feature Selection and Justification
def select_features(df_encoded):
    """Select and justify features for classification"""
    # Calculate feature importance with a simple decision tree
    X = df_encoded.drop('class', axis=1)
    y = df_encoded['class']
    
    feature_names = X.columns
    
    # Train a decision tree to get feature importances
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X, y)
    
    # Get feature importances
    importances = dt.feature_importances_
    
    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance for Mushroom Classification')
    plt.tight_layout()
    plt.show()
    
    # Define different feature sets to compare
    # Case 1: Top feature only
    top_feature = feature_importance_df['Feature'].iloc[0]
    
    # Case 2: Top 3 features
    top_3_features = feature_importance_df['Feature'].iloc[:3].tolist()
    
    # Case 3: Top 5 features
    top_5_features = feature_importance_df['Feature'].iloc[:5].tolist()
    
    # Case 4: All features
    all_features = feature_names.tolist()
    
    feature_sets = {
        'top_feature': [top_feature],
        'top_3_features': top_3_features,
        'top_5_features': top_5_features,
        'all_features': all_features
    }
    
    print("Selected feature sets:")
    for name, features in feature_sets.items():
        print(f"{name}: {features}")
    
    return feature_sets

# Section 4: Train Models
def train_and_evaluate_models(df_encoded, feature_sets):
    """Train and evaluate different classification models on various feature sets"""
    X = df_encoded.drop('class', axis=1)
    y = df_encoded['class']
    
    # Define models to test
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Dictionary to store results
    results = {}
    
    for feature_set_name, features in feature_sets.items():
        print(f"\nEvaluating models with feature set: {feature_set_name}")
        X_subset = X[features]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=0.2, random_state=42, stratify=y
        )
        
        feature_results = {}
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Store results
            feature_results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }
            
            print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                  f"Recall: {recall:.4f}, F1: {f1:.4f}")
        
        results[feature_set_name] = feature_results
    
    # Find best models
    best_accuracy = 0
    best_model_info = None
    
    for feature_set, models_dict in results.items():
        for model_name, metrics in models_dict.items():
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_model_info = {
                    'feature_set': feature_set,
                    'model_name': model_name,
                    'metrics': metrics
                }
    
    print(f"\nBest model: {best_model_info['model_name']} with {best_model_info['feature_set']}")
    print(f"Accuracy: {best_model_info['metrics']['accuracy']:.4f}")
    
    return results, best_model_info

# Section 5: Improve Models
def tune_best_model(best_model_info):
    """Tune the hyperparameters of the best model"""
    model_name = best_model_info['model_name']
    feature_set = best_model_info['feature_set']
    metrics = best_model_info['metrics']
    
    X_train = metrics['X_train']
    y_train = metrics['y_train']
    X_test = metrics['X_test']
    y_test = metrics['y_test']
    
    print(f"Tuning {model_name} with {feature_set}...")
    
    # Define parameter grids for each model type
    if model_name == 'Decision Tree':
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = DecisionTreeClassifier(random_state=42)
        
    elif model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = RandomForestClassifier(random_state=42)
        
    elif model_name == 'Logistic Regression':
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['liblinear', 'saga']
        }
        base_model = LogisticRegression(random_state=42, max_iter=2000)
    
    # Perform grid search with cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Tuned model - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
          f"Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Compare with baseline model
    baseline_accuracy = best_model_info['metrics']['accuracy']
    improvement = (accuracy - baseline_accuracy) / baseline_accuracy * 100
    
    print(f"Improvement over baseline: {improvement:.2f}%")
    
    # Confusion matrix for the best model
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Edible', 'Poisonous'],
                yticklabels=['Edible', 'Poisonous'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Tuned Model')
    plt.tight_layout()
    plt.show()
    
    return best_model, grid_search.best_params_

# Section 6: Final Thoughts & Insights
def summarize_findings(df, df_encoded, best_model_info, tuned_model, best_params, label_encoders):
    """Summarize findings and provide insights"""
    print("\n--- FINAL SUMMARY ---")
    print(f"Dataset contains {df.shape[0]} mushroom samples with {df.shape[1]} features")
    print(f"Class distribution: {df['class'].value_counts().to_dict()}")
    
    print(f"\nBest model: {best_model_info['model_name']} using {best_model_info['feature_set']}")
    print(f"Best hyperparameters: {best_params}")
    
    # If the best model is a tree-based model, show feature importance
    if best_model_info['model_name'] in ['Decision Tree', 'Random Forest']:
        features = best_model_info['feature_set']
        importances = tuned_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance in Tuned Model')
        plt.tight_layout()
        plt.show()
        
        print("\nMost important features:")
        for idx, row in importance_df.iterrows():
            feature = row['Feature']
            importance = row['Importance']
            # Get original feature values (before encoding)
            original_values = df[feature].unique()
            encoded_values = {label_encoders[feature].transform([val])[0]: val for val in original_values}
            
            print(f"{feature}: {importance:.4f}")
            print(f"  Original values: {', '.join(original_values)}")
            print(f"  Encoded mapping: {encoded_values}")
    
    print("\nClassification report:")
    y_pred = tuned_model.predict(best_model_info['metrics']['X_test'])
    print(classification_report(best_model_info['metrics']['y_test'], y_pred, 
                              target_names=['Edible', 'Poisonous']))
    
    print("\nInsights and recommendations:")
    print("1. The most predictive features for mushroom edibility are...")
    print("2. The model achieved high accuracy, indicating...")
    print("3. For practical applications, consider...")
    print("4. Limitations of this analysis include...")
    print("5. Future work could include...")

# Additional code to improve model performance
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score, learning_curve

def advanced_modeling(df_encoded, best_model_info):
    """Apply advanced modeling techniques to improve performance"""
    
    # Extract best feature set and data
    feature_set = best_model_info['feature_set']
    X = df_encoded[feature_set]
    y = df_encoded['class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Applying advanced modeling techniques...")
    
    # 1. Create an ensemble of different classifiers
    estimators = [
        ('rf', RandomForestClassifier(random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ]
    
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    
    # Train and evaluate ensemble
    ensemble.fit(X_train, y_train)
    ensemble_pred = ensemble.predict(X_test)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    
    print(f"Ensemble model accuracy: {ensemble_acc:.4f}")
    
    # 2. Feature selection with SelectFromModel
    selector = SelectFromModel(
        GradientBoostingClassifier(random_state=42), threshold='median'
    )
    selector.fit(X_train, y_train)
    
    # Get selected features
    selected_features = X.columns[selector.get_support()]
    print(f"Features selected by automatic selection: {selected_features.tolist()}")
    
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # Train a model on selected features
    rf_selected = RandomForestClassifier(random_state=42)
    rf_selected.fit(X_train_selected, y_train)
    selected_pred = rf_selected.predict(X_test_selected)
    selected_acc = accuracy_score(y_test, selected_pred)
    
    print(f"Model with automatic feature selection accuracy: {selected_acc:.4f}")
    
    # 3. Learning curves to detect overfitting/underfitting
    plt.figure(figsize=(10, 6))
    train_sizes, train_scores, test_scores = learning_curve(
        RandomForestClassifier(random_state=42), X_train, y_train, 
        train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    
    plt.xlabel('Training examples')
    plt.ylabel('Accuracy score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    
    # 4. ROC Curve for binary classification
    best_model = best_model_info['metrics']['model']
    probas = best_model.predict_proba(X_test)
    
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    return {
        'ensemble': {'model': ensemble, 'accuracy': ensemble_acc},
        'feature_selection': {'model': rf_selected, 'features': selected_features, 'accuracy': selected_acc}
    }


def test_feature_combinations(df_encoded):
    """Test different combinations of features for performance"""
    # Get list of all features
    features = df_encoded.columns.tolist()
    features.remove('class')
    
    # Define some interesting combinations to test
    combinations = [
        # Group by type
        ['cap-shape', 'cap-surface', 'cap-color'],  # Cap features
        ['gill-attachment', 'gill-spacing', 'gill-size', 'gill-color'],  # Gill features
        ['stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
         'stalk-color-above-ring', 'stalk-color-below-ring'],  # Stalk features
        ['odor'],  # Smell features (often very predictive for mushrooms)
        
        # Specific combinations that might be easy to observe in the wild
        ['cap-color', 'gill-color', 'odor'],
        ['cap-shape', 'cap-color', 'ring-type', 'habitat']
    ]
    
    # Add names for each combination
    named_combinations = {
        'cap_features': combinations[0],
        'gill_features': combinations[1],
        'stalk_features': combinations[2],
        'odor_only': combinations[3],
        'visual_and_smell': combinations[4],
        'field_identifiable': combinations[5]
    }
    
    # Train a basic random forest on each combination
    results = {}
    X = df_encoded.drop('class', axis=1)
    y = df_encoded['class']
    
    for name, feature_list in named_combinations.items():
        print(f"\nTesting feature combination: {name}")
        X_subset = X[feature_list]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'features': feature_list,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"Accuracy: {accuracy:.4f}")
    
    # Show results in a nice table
    results_df = pd.DataFrame({
        name: {
            'Accuracy': info['accuracy'],
            'Precision': info['precision'],
            'Recall': info['recall'],
            'F1 Score': info['f1'],
            'Num Features': len(info['features'])
        } for name, info in results.items()
    }).T.sort_values('Accuracy', ascending=False)
    
    print("\nFeature Combination Results:")
    print(results_df)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    sns.heatmap(results_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']], 
                annot=True, cmap='YlGnBu')
    plt.title('Performance Metrics for Different Feature Combinations')
    plt.tight_layout()
    plt.show()
    
    return results, results_df

def visualize_hyperparameter_tuning(X, y, param_name, param_range, model_class, fixed_params=None):
    """Visualize the effect of a hyperparameter on model performance"""
    if fixed_params is None:
        fixed_params = {}
    
    train_scores = []
    test_scores = []
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    for param_value in param_range:
        # Set up model with this parameter value
        params = {param_name: param_value, 'random_state': 42, **fixed_params}
        model = model_class(**params)
        
        # Train and evaluate
        model.fit(X_train, y_train)
        
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
        
        train_scores.append(train_accuracy)
        test_scores.append(test_accuracy)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_scores, 'o-', label='Training Accuracy')
    plt.plot(param_range, test_scores, 'o-', label='Testing Accuracy')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.title(f'Effect of {param_name} on Model Performance')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Find optimal value
    best_idx = np.argmax(test_scores)
    best_param = param_range[best_idx]
    best_score = test_scores[best_idx]
    
    print(f"Best {param_name} value: {best_param}")
    print(f"Test accuracy with this value: {best_score:.4f}")
    
    return best_param, best_score

# Full analysis
df = load_and_inspect_data('agaricus-lepiota-sample.data')
df, df_encoded, label_encoders = explore_and_prepare_data(df)
feature_sets = select_features(df_encoded)

# Run the standard models first
results, best_model_info = train_and_evaluate_models(df_encoded, feature_sets)
tuned_model, best_params = tune_best_model(best_model_info)

# Try alternative feature combinations
feature_results, feature_results_df = test_feature_combinations(df_encoded)

# Apply advanced modeling techniques
advanced_results = advanced_modeling(df_encoded, best_model_info)

# Visualize hyperparameter tuning for a key parameter
X = df_encoded[best_model_info['feature_set']]
y = df_encoded['class']

if best_model_info['model_name'] == 'Random Forest':
    visualize_hyperparameter_tuning(
        X, y, 'n_estimators', [10, 50, 100, 200, 300, 500], 
        RandomForestClassifier
    )
elif best_model_info['model_name'] == 'Decision Tree':
    visualize_hyperparameter_tuning(
        X, y, 'max_depth', [None, 5, 10, 15, 20, 30], 
        DecisionTreeClassifier
    )

# Final summary
summarize_findings(df, df_encoded, best_model_info, tuned_model, best_params, label_encoders)

# Main function
def main():
    # Load and inspect data
    df = load_and_inspect_data('agaricus-lepiota-sample.data')
    
    # Explore and prepare data
    df, df_encoded, label_encoders = explore_and_prepare_data(df)
    
    # Select features
    feature_sets = select_features(df_encoded)
    
    # Train and evaluate models
    results, best_model_info = train_and_evaluate_models(df_encoded, feature_sets)
    
    # Tune best model
    tuned_model, best_params = tune_best_model(best_model_info)
    
    # Summarize findings
    summarize_findings(df, df_encoded, best_model_info, tuned_model, best_params, label_encoders)

if __name__ == "__main__":
    main()