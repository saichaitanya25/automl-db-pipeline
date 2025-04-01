# score_batch.py
# Batch inference using the best AutoML model

from pyspark.sql import SparkSession
import mlflow
from mlflow.pyfunc import load_model

spark = SparkSession.builder.appName("BatchScoring").getOrCreate()

# Load test data
test_df = spark.read.format("delta").load("/mnt/delta/test_data")

# Load the latest model from MLflow registry
model = load_model("models:/AutoML_Best_Model/Production")

# Apply the model
predictions = model.predict(test_df.toPandas())

# Optionally convert and save predictions
pred_df = spark.createDataFrame(predictions)
pred_df.write.mode("overwrite").format("delta").save("/mnt/delta/predictions")
