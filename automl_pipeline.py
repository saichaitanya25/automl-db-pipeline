# automl_pipeline.py
# AutoML pipeline using Databricks, Delta Lake, and MLflow

import databricks.automl
from pyspark.sql import SparkSession
import mlflow
from databricks.automl.runtime import score

# Start Spark session
spark = SparkSession.builder.appName("AutoMLPipeline").getOrCreate()

# Load data from Delta table or file
df = spark.read.format("delta").load("/mnt/delta/training_data")

# Split into train/test
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Launch AutoML (classification task example)
summary = databricks.automl.classify(train_df, target_col="target", timeout_minutes=15)

# Register best model to MLflow
best_model_run_id = summary.best_trial.mlflow_run_id
model_uri = f"runs:/{best_model_run_id}/model"
mlflow.register_model(model_uri, "AutoML_Best_Model")
