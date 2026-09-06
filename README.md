# MLflow
MLOPS-Experiments-with-MLFlow

MLflow is an open-source MLOps framework used to manage the end-to-end machine learning lifecycle.
It helps you track experiments, manage models, and make ML work reproducible.

In simple terms:

MLflow helps you answer the question:
“Which model did we train, with what parameters, and how good was it?”



For running MLflow in remote server so that all other team can access that and see the experiments for that we can use Dagshub or any cloud services.
 - For dagshub : pip install dagshub in terminal
 - Go to dagshub website and signup and connect your repo. Once connected In experiment section you will find the remote link of MLflow server and the code to import. So put that in the existing python file.  
 - We can directly the register the model in mlflow and tag it like which particular stage(Dev, Stage, Prod) it is in.
