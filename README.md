# Telco_Churn_Case_Study

## Overview
High level thought process shared here - 
[click here](https://docs.google.com/document/d/1Z5gmKob3s0WiOLm0H9JuCg4YTE8fAAWZjg62SNFy4xA/edit?addon_store)

## Installation

Install Apache Airflow


```pip install apache-airflow```

Initialize the Airflow Database

```airflow db init```

Create a Directory for Airflow DAGs

```mkdir -p ~/airflow/dags```

Clone the Repository

```
git clone https://github.com/DisTurbed-afk/Telco_Churn_Case_Study.git
cd Telco_Churn_Case_Study
```

Copy the DAG to the Airflow DAGs Directory

```cp airflow/dags/churn_prediction_dag.py ~/airflow/dags/```

Install Required Python Packages

```pip install pandas scikit-learn xgboost joblib matplotlib seaborn```

Start Scheduler and Webserver

```
airflow scheduler
airflow webserver
```

http://localhost:8080 - > can monitor and trigger the dag