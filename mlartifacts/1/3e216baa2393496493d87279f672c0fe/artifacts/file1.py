import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load Wine dataset
wine = load_wine()
x = wine.data
y = wine.target

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

# Define the params for RF(RandomForest) model
#Random Forest is an ensemble method that uses multiple decision trees to improve classification accuracy.
max_depth = 10 #This is the maximum depth of the tree
n_estimators = 20 #This is the number of trees in the forest


#Mention your experiment details here
mlflow.set_experiment("Test") #or we can do the same inside mlflow.start_run(experiment_id="Test") and this both can create a new experiment if it doesn't exist.

# MLflow Configuration
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42) #This line initializes the RandomForestClassifier with specified parameters
    rf.fit(x_train, y_train) #This line trains the model on the training data
    y_pred = rf.predict(x_test) #This line makes predictions on the test data
    accuracy = accuracy_score(y_test, y_pred) #This line calculates the accuracy of the model
   
    mlflow.log_metric("accuracy", accuracy)

    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

 # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig("Confusion-matrix.png")

    # log artifacts using mlflow
    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)


    # Tags
    mlflow.set_tag({"Author": "Rajesh", "Project": "Wine Quality Prediction"})
    # Log the model
    mlflow.sklearn.log_model(rf, "model")

    print(f"Accuracy: {accuracy}")


